"""
绩效分析引擎 — 多维度量化评估

指标列表：
  - total_return:   总收益率
  - annual_return:  年化收益率
  - max_drawdown:   最大回撤
  - sharpe_ratio:   夏普比率
  - calmar_ratio:   卡玛比率 (年化收益 / 最大回撤)
  - information_ratio: 信息比率 (超额收益 / 跟踪误差)
  - win_rate:       胜率
  - profit_factor:  盈亏比
  - annual_volatility: 年化波动率
  - benchmark_return:  基准收益率
  - excess_return:     超额收益
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 假设每年约 244 个交易日
TRADING_DAYS_PER_YEAR = 244


def compute_metrics(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.02,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    计算完整绩效指标

    Args:
        equity_curve: 权益曲线 (index=date, values=portfolio value)
        trades: 交易记录 (VectorBT trades.records_readable)
        risk_free_rate: 无风险利率 (默认 2%)
        benchmark_returns: 基准日收益率序列

    Returns:
        Dict of metric_name -> value
    """
    metrics = {}

    # ---- 入口规范化: 确保 equity_curve 是 1D pd.Series ----
    if isinstance(equity_curve, pd.DataFrame):
        # VectorBT 可能返回多列 DataFrame，取均值压平
        if equity_curve.shape[1] == 1:
            equity_curve = equity_curve.iloc[:, 0]
        else:
            equity_curve = equity_curve.mean(axis=1)
    # 确保是 Series 类型
    if not isinstance(equity_curve, pd.Series):
        equity_curve = pd.Series(equity_curve)

    if equity_curve.empty or len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "information_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "annual_volatility": 0.0,
            "benchmark_return": 0.0,
            "excess_return": 0.0,
            "total_days": len(equity_curve),
        }

    # ---- 基础指标 ----
    start_val = float(equity_curve.iloc[0])
    end_val = float(equity_curve.iloc[-1])
    total_return = (end_val / start_val) - 1.0
    metrics["total_return"] = round(total_return, 6)

    total_days = len(equity_curve)
    years = total_days / TRADING_DAYS_PER_YEAR

    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return / max(years, 0.01)
    metrics["annual_return"] = round(annual_return, 6)

    # ---- 日收益率序列 ----
    daily_returns = equity_curve.pct_change().dropna()

    # ---- 年化波动率 ----
    daily_vol = float(daily_returns.std())
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    metrics["annual_volatility"] = round(float(annual_vol), 6)

    # ---- 最大回撤 ----
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_dd = float(drawdown.min())
    metrics["max_drawdown"] = round(max_dd, 6)

    # ---- 夏普比率 ----
    if daily_vol > 0:
        excess_daily = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
        sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * (excess_daily.mean() / daily_vol)
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = round(sharpe, 4)

    # ---- 卡玛比率 (Calmar Ratio) ----
    if abs(max_dd) > 1e-8:
        calmar = annual_return / abs(max_dd)
    else:
        calmar = 0.0
    metrics["calmar_ratio"] = round(calmar, 4)

    # ---- 信息比率 vs 基准 ----
    benchmark_return = 0.0
    excess_return = 0.0
    ir = 0.0

    if benchmark_returns is not None and not benchmark_returns.empty:
        # 对齐日期
        aligned = pd.concat(
            [daily_returns, benchmark_returns], axis=1
        ).dropna()
        aligned.columns = ["strategy", "benchmark"]

        benchmark_return = (1 + aligned["benchmark"]).prod() - 1
        excess_returns = aligned["strategy"] - aligned["benchmark"]
        excess_return = (1 + excess_returns).prod() - 1
        tracking_error = excess_returns.std()

        if years > 0 and tracking_error > 1e-8:
            ir = (excess_returns.mean() / tracking_error) * np.sqrt(TRADING_DAYS_PER_YEAR)
        elif tracking_error > 1e-8:
            ir = excess_returns.mean() / tracking_error

    metrics["benchmark_return"] = round(benchmark_return, 6)
    metrics["excess_return"] = round(excess_return, 6)
    metrics["information_ratio"] = round(ir, 4)

    # ---- 交易统计 (如果提供了交易记录) ----
    if trades is not None and not trades.empty:
        # 胜率
        pnl_col = None
        for col in ["PnL", "pnl", "Profit", "Return"]:
            if col in trades.columns:
                pnl_col = col
                break

        if pnl_col is not None:
            win_rate = (trades[pnl_col] > 0).mean()
            metrics["win_rate"] = round(win_rate, 4)

            # 盈亏比
            winning = trades.loc[trades[pnl_col] > 0, pnl_col]
            losing = trades.loc[trades[pnl_col] < 0, pnl_col]

            avg_win = winning.mean() if not winning.empty else 0
            avg_loss = abs(losing.mean()) if not losing.empty else 1

            if avg_loss > 1e-8:
                profit_factor = avg_win / avg_loss
            else:
                profit_factor = float("inf") if avg_win > 0 else 0.0

            metrics["profit_factor"] = round(profit_factor, 4)
        else:
            metrics["win_rate"] = 0.0
            metrics["profit_factor"] = 0.0
    else:
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0

    metrics["total_days"] = total_days
    metrics["years"] = round(years, 2)

    return metrics


def format_metrics_report(
    metrics: Dict[str, float],
    strategy_name: str = "Unknown",
) -> str:
    """
    格式化绩效报告（控制台输出）

    Args:
        metrics: compute_metrics 返回的指标字典
        strategy_name: 策略名称

    Returns:
        格式化的报告字符串
    """
    report = f"""
╔══════════════════════════════════════════════════╗
║            📈 绩效报告 — {strategy_name[:30]:30s} ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  总收益率       │ {metrics.get('total_return', 0)*100:>10.2f}%           ║
║  年化收益率     │ {metrics.get('annual_return', 0)*100:>10.2f}%           ║
║  基准收益率     │ {metrics.get('benchmark_return', 0)*100:>10.2f}%           ║
║  超额收益       │ {metrics.get('excess_return', 0)*100:>10.2f}%           ║
║  最大回撤       │ {metrics.get('max_drawdown', 0)*100:>10.2f}%           ║
║  年化波动率     │ {metrics.get('annual_volatility', 0)*100:>10.2f}%           ║
║  夏普比率       │ {metrics.get('sharpe_ratio', 0):>10.2f}               ║
║  卡玛比率       │ {metrics.get('calmar_ratio', 0):>10.2f}               ║
║  信息比率       │ {metrics.get('information_ratio', 0):>10.2f}               ║
║  胜率           │ {metrics.get('win_rate', 0)*100:>10.2f}%           ║
║  盈亏比         │ {metrics.get('profit_factor', 0):>10.2f}               ║
║                                                  ║
║  回测区间       │ {metrics.get('total_days', 0)} 天 / {metrics.get('years', 0)} 年    ║
╚══════════════════════════════════════════════════╝
"""
    return report


def metrics_to_feedback_context(
    metrics: Dict[str, float],
    strategy_name: str = "Unknown",
    strategy_version: int = 1,
) -> str:
    """
    将指标转为反馈给 LLM 的文本上下文

    用于进化循环中的「反思反馈」步骤。

    Args:
        metrics: 绩效指标
        strategy_name: 策略名称
        strategy_version: 策略版本号

    Returns:
        LLM 可读的文本上下文
    """
    total_return = metrics.get("total_return", 0) * 100
    annual_return = metrics.get("annual_return", 0) * 100
    max_dd = metrics.get("max_drawdown", 0) * 100
    sharpe = metrics.get("sharpe_ratio", 0)
    calmar = metrics.get("calmar_ratio", 0)
    ir = metrics.get("information_ratio", 0)
    win_rate = metrics.get("win_rate", 0) * 100
    excess = metrics.get("excess_return", 0) * 100

    # 生成弱点分析
    issues = []
    if annual_return < 0:
        issues.append("- ⚠️ 策略年化收益率为负，整体选股逻辑可能失效")
    if max_dd > 0.20:
        issues.append(f"- ⚠️ 最大回撤达到 {max_dd:.1f}%，需要增加防守型因子或止损机制")
    if sharpe < 0.5:
        issues.append(f"- ⚠️ 夏普比率仅 {sharpe:.2f}，风险调整后收益不足")
    if calmar < 0.5:
        issues.append(f"- ⚠️ 卡玛比率仅 {calmar:.2f}，回撤控制能力较弱")
    if win_rate < 40:
        issues.append(f"- ⚠️ 胜率仅 {win_rate:.1f}%，选股命中率偏低")
    if excess < 0:
        issues.append(f"- ⚠️ 超额收益为负，策略未能跑赢基准")

    issues_text = "\n".join(issues) if issues else "- ✅ 策略整体表现尚可"

    context = f"""📊 回测结果 — {strategy_name} (v{strategy_version})

【核心指标】
  总收益率: {total_return:.2f}%
  年化收益率: {annual_return:.2f}%
  最大回撤: {max_dd:.2f}%
  夏普比率: {sharpe:.2f}
  卡玛比率: {calmar:.2f}
  信息比率: {ir:.2f}
  胜率: {win_rate:.1f}%
  超额收益: {excess:.2f}%

【诊断分析】
{issues_text}

请根据以上结果，分析策略的问题并给出改进方案，生成下一版策略代码。
"""
    return context


def save_metrics_json(
    metrics: Dict[str, float],
    filepath: str = "results/latest.json",
) -> None:
    """将指标保存为 JSON 文件"""
    import json

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保所有值可 JSON 序列化
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            clean_metrics[k] = float(v)
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            clean_metrics[k] = 0.0
        else:
            clean_metrics[k] = v

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2, ensure_ascii=False)

    print(f"💾 指标已保存: {out_path}")