"""
回测引擎 — 基于 VectorBT 的高速向量化回测

功能：
  - 接收策略输出的信号 DataFrame
  - 构建 VectorBT Portfolio
  - 输出权益曲线、交易记录
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False


def prepare_price_matrix(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    将长格式数据转为 VectorBT 所需的宽格式价格矩阵

    Args:
        data: 长格式 DataFrame (date, code, close, ...)

    Returns:
        pd.DataFrame: rows=date, cols=code, values=close
    """
    pivot = data.pivot_table(
        index="date",
        columns="code",
        values="close",
        aggfunc="last",
    )
    # 前向填充停牌日
    pivot = pivot.ffill()
    return pivot


def prepare_signal_matrix(
    signals: pd.DataFrame,
    price_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    将信号 DataFrame 转换为 VectorBT 的信号矩阵

    VectorBT 的 targetpercent 需要每天都有目标权重值。
    我们在调仓日设置目标权重，然后用前向填充保持持仓。

    Args:
        signals: columns [code, weight, date]
        price_matrix: 价格矩阵 (rows=date, cols=code)

    Returns:
        pd.DataFrame: 调仓日权重矩阵（NaN=不变，VectorBT 会保持当前持仓）
    """
    # 初始化 NaN 矩阵（NaN 表示无调仓动作，保持当前持仓）
    signal_matrix = pd.DataFrame(
        np.nan,
        index=price_matrix.index,
        columns=price_matrix.columns,
    )

    # 按调仓日期填充目标权重
    for _, row in signals.iterrows():
        code = row["code"]
        dt = row["date"]
        weight = row["weight"]

        # 找到最近的交易日（如果调仓日不在交易日历中）
        if dt not in signal_matrix.index:
            valid_dates = signal_matrix.index[signal_matrix.index <= dt]
            if len(valid_dates) == 0:
                continue
            dt = valid_dates[-1]

        if code in signal_matrix.columns:
            signal_matrix.loc[dt, code] = weight

    return signal_matrix


def run_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    commission: float = 0.0003,
    slippage: float = 0.001,
) -> Dict:
    """
    执行 VectorBT 回测

    Args:
        data: 日线数据 DataFrame
        signals: 策略信号 [code, weight, date]
        initial_capital: 初始资金
        commission: 手续费率
        slippage: 滑点

    Returns:
        Dict:
            - equity_curve: pd.Series — 权益曲线
            - trades: pd.DataFrame — 交易记录
            - stats: pd.Series — 回测统计
            - portfolio: vbt.Portfolio — VectorBT 原始对象
    """
    if not HAS_VBT:
        raise ImportError(
            "vectorbt 未安装，请运行: pip install vectorbt"
        )

    print("🔬 准备回测数据...")

    # 构建价格矩阵
    price_matrix = prepare_price_matrix(data)
    print(f"   价格矩阵: {price_matrix.shape[0]} 交易日 × {price_matrix.shape[1]} 股票")

    # 构建信号矩阵
    signal_matrix = prepare_signal_matrix(signals, price_matrix)

    # 按权重比例的买入信号
    print("📊 执行向量化回测...")

    # 使用 vbt.Portfolio.from_orders with targetpercent
    # signal_matrix: NaN=保持持仓, 0.0~1.0=目标权重
    # 将 signal_matrix 中的 NaN 替换为 0，VectorBT 在 targetpercent 模式下
    # NaN 不会被当作订单，只有显式设置的权重才会触发调仓
    portfolio = vbt.Portfolio.from_orders(
        close=price_matrix,
        size=signal_matrix,
        size_type="targetpercent",
        init_cash=initial_capital,
        fees=commission,
        slippage=slippage,
        freq="1D",
    )

    # 提取结果
    equity_curve = portfolio.value()
    # 如果 equity_curve 是 DataFrame，取均值整合为 Series
    if isinstance(equity_curve, pd.DataFrame):
        if equity_curve.shape[1] == 1:
            equity_curve = equity_curve.iloc[:, 0]
        else:
            equity_curve = equity_curve.mean(axis=1)
    trades = portfolio.trades.records_readable
    stats = portfolio.stats() if hasattr(portfolio, 'stats') else pd.Series()

    print(f"✅ 回测完成 — 总交易次数: {len(trades)}")

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "stats": stats,
        "portfolio": portfolio,
    }


def run_backtest_simple(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    commission: float = 0.0003,
    slippage: float = 0.001,
) -> Dict:
    """
    简化版回测入口（容错更强）

    用于处理 AI 生成的可能不太标准的信号。
    """
    if signals is None or signals.empty:
        return {
            "error": "信号为空，无法回测",
            "equity_curve": pd.Series(dtype=float),
            "trades": pd.DataFrame(),
            "stats": pd.Series(dtype=object),
        }

    # 确保 signals 格式正确
    signals = signals.copy()
    if "code" not in signals.columns or "date" not in signals.columns:
        return {
            "error": "信号缺少 code/date 列",
            "equity_curve": pd.Series(dtype=float),
            "trades": pd.DataFrame(),
            "stats": pd.Series(dtype=object),
        }

    signals["weight"] = pd.to_numeric(signals.get("weight", 1.0), errors="coerce")
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    signals["code"] = signals["code"].astype(str).str.strip()
    signals = signals.dropna(subset=["code", "date", "weight"])

    if signals.empty:
        return {
            "error": "信号为空或不合法，无法回测",
            "equity_curve": pd.Series(dtype=float),
            "trades": pd.DataFrame(),
            "stats": pd.Series(dtype=object),
        }

    if signals["weight"].isna().all():
        signals["weight"] = 1.0 / signals.groupby("date")["code"].transform("count")

    signals = (
        signals.groupby(["date", "code"], as_index=False)["weight"]
        .mean()
        .sort_values(["date", "weight"], ascending=[True, False])
    )

    try:
        result = run_backtest(
            data=data,
            signals=signals,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
        )
        return result
    except Exception as e:
        return {
            "error": str(e),
            "equity_curve": pd.Series(dtype=float),
            "trades": pd.DataFrame(),
            "stats": pd.Series(dtype=object),
        }