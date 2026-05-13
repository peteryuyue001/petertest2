#!/usr/bin/env python3
"""
============================================================
  🧬 Quant Evolution — 非交互式自动化运行脚本
============================================================

解决 main.py 中 input() 导致 Cline/CI 环境卡住的问题。
本脚本完全无需用户输入，可直接被 Cline 调用。

用法:
  python3 run_auto.py --step fetch          # 下载数据
  python3 run_auto.py --step generate       # 生成策略
  python3 run_auto.py --step backtest       # 回测最新策略
  python3 run_auto.py --step backtest --strategy strategy_v8.py  # 回测指定策略
  python3 run_auto.py --step backtest_all   # 回测所有策略
  python3 run_auto.py --step report         # 查看绩效报告
  python3 run_auto.py --step evolve         # AI 改进策略
  python3 run_auto.py --step history        # 查看进化历史
  python3 run_auto.py --step full_cycle     # 完整进化循环（fetch→generate→backtest→evolve）
  python3 run_auto.py --step evolve_loop --n 3  # 连续进化 N 轮
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ============================================================
# 配置加载
# ============================================================

def load_config():
    try:
        import config
        return config
    except ImportError:
        import importlib.util
        example_path = PROJECT_ROOT / "config.example.py"
        spec = importlib.util.spec_from_file_location("config_example", str(example_path))
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        return cfg


# ============================================================
# Step 1: 下载数据（无交互）
# ============================================================

def auto_fetch(cfg):
    print("\n" + "=" * 60)
    print("  📊 Step 1: 下载/更新 A股数据（自动模式）")
    print("=" * 60)

    from data.fetcher import DataFetcher

    fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)
    df = fetcher.fetch_daily(
        stock_pool=cfg.STOCK_POOL,
        start_date=cfg.DATA_START_DATE,
        end_date=cfg.DATA_END_DATE,
    )
    print(f"\n✅ 数据就绪: {len(df)} 行, {df['code'].nunique()} 支股票")

    # 验证数据质量
    import numpy as np
    daily_ret = df.groupby('date')['pct_change'].mean()
    annual_vol = daily_ret.std() * np.sqrt(244)
    print(f"   年化波动率估算: {annual_vol:.2f}%")
    if annual_vol < 5:
        print("   ⚠️  警告：年化波动率过低，可能是合成数据！")
        print("   💡 建议：检查网络连接，确保 AkShare 能访问真实数据")
    else:
        print("   ✅ 数据质量正常（真实 A 股数据）")

    return fetcher


# ============================================================
# Step 2: 生成策略（无交互）
# ============================================================

def auto_generate(cfg, instruction: str = ""):
    print("\n" + "=" * 60)
    print("  🧠 Step 2: DeepSeek 生成策略（自动模式）")
    print("=" * 60)

    from llm.deepseek_client import create_client_from_config

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n❌ {e}")
        return None

    if not instruction:
        instruction = (
            "生成一个 A 股沪深300多因子选股策略。"
            "包含以下因子：20日动量、波动率过滤、换手率因子。"
            "月度调仓，持仓 20 支，等权重。"
            "策略必须能跑赢沪深300基准，夏普比率目标 > 0.5。"
        )

    print(f"\n🤖 正在调用 DeepSeek 生成策略...")
    print(f"   指令: {instruction[:100]}...")

    code, error = client.generate_strategy(instruction=instruction)

    if error:
        print(f"\n❌ 策略生成失败: {error}")
        return None

    if not code:
        print("\n❌ 策略生成返回空内容")
        return None

    # 确定版本号
    pool_dir = PROJECT_ROOT / "strategy_pool"
    existing = list(pool_dir.glob("strategy_v*.py"))
    versions = []
    for f in existing:
        try:
            v = int(f.stem.replace("strategy_v", ""))
            versions.append(v)
        except ValueError:
            continue
    version = max(versions) + 1 if versions else 1

    strategy_path = pool_dir / f"strategy_v{version}.py"
    strategy_path.write_text(code, encoding="utf-8")

    print(f"\n✅ 策略已保存: strategy_pool/strategy_v{version}.py")
    lines = code.split("\n")[:20]
    print("\n--- 策略代码预览 (前 20 行) ---")
    for line in lines:
        print(f"  {line}")

    return str(strategy_path)


# ============================================================
# Step 3: 回测（无交互）
# ============================================================

def auto_backtest(cfg, strategy_name: str = "", fetcher=None):
    print("\n" + "=" * 60)
    print("  🔬 Step 3: 运行回测（自动模式）")
    print("=" * 60)

    import pandas as pd
    from data.fetcher import DataFetcher
    from engine.sandbox import execute_strategy, format_error_for_llm
    from engine.backtest import run_backtest_simple
    from engine.analyzer import compute_metrics, save_metrics_json, format_metrics_report
    from engine.sandbox import load_strategy_from_file

    pool_dir = PROJECT_ROOT / "strategy_pool"

    if strategy_name:
        target = pool_dir / strategy_name
        if not target.exists():
            print(f"❌ 策略文件不存在: {target}")
            return None
    else:
        strategies = sorted(pool_dir.glob("strategy_v*.py"))
        if not strategies:
            print("❌ 没有找到策略文件！请先运行 --step generate")
            return None
        target = strategies[-1]

    print(f"\n🔬 回测: {target.name}")

    # 获取数据
    if fetcher is None:
        fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)

    data = fetcher.get_data(
        start_date=cfg.DATA_START_DATE,
        end_date=cfg.DATA_END_DATE,
        universe=cfg.STOCK_POOL,
    )

    # 沙箱执行策略
    print("   📦 沙箱执行策略...")
    signals, error = execute_strategy(target, data)

    if error:
        print(f"\n❌ 策略执行失败:\n   {error}")

        # 自动尝试 AI 修复（无需用户确认）
        print("\n🔧 自动触发 AI 修复...")
        try:
            from llm.deepseek_client import create_client_from_config
            client = create_client_from_config()
            code = target.read_text(encoding="utf-8")
            fixed_code, fix_error = client.fix_code(
                error_message=error,
                strategy_code=code,
            )
            if fixed_code and not fix_error:
                existing = list(pool_dir.glob("strategy_v*.py"))
                versions = [int(f.stem.replace("strategy_v", "")) for f in existing
                            if f.stem.replace("strategy_v", "").isdigit()]
                version = max(versions) + 1 if versions else 1
                new_path = pool_dir / f"strategy_v{version}.py"
                new_path.write_text(fixed_code, encoding="utf-8")
                print(f"   ✅ 修复代码已保存: {new_path.name}")
                print(f"   💡 请重新运行 --step backtest 测试修复后的策略")
            else:
                print(f"   ❌ AI 修复失败: {fix_error}")
        except Exception as e:
            print(f"   ❌ 无法连接 LLM: {e}")
        return None

    # 运行回测
    print("   📊 执行 VectorBT 回测...")
    result = run_backtest_simple(
        data=data,
        signals=signals,
        initial_capital=cfg.INITIAL_CAPITAL,
        commission=cfg.COMMISSION,
        slippage=cfg.SLIPPAGE,
    )

    if "error" in result and result["error"]:
        print(f"   ❌ 回测失败: {result['error']}")
        return None

    # 获取基准收益
    benchmark_returns = None
    try:
        benchmark = fetcher.fetch_benchmark(
            symbol=cfg.BENCHMARK_SYMBOL,
            start_date=cfg.DATA_START_DATE,
            end_date=cfg.DATA_END_DATE,
        )
        benchmark_returns = benchmark.set_index("date")["pct_change"].dropna()
    except Exception:
        pass

    # 计算指标
    equity = result.get("equity_curve", pd.Series(dtype=float))
    trades = result.get("trades")

    metrics = compute_metrics(
        equity_curve=equity,
        trades=trades,
        risk_free_rate=cfg.RISK_FREE_RATE,
        benchmark_returns=benchmark_returns,
    )

    # 读取策略元信息
    strategy, _ = load_strategy_from_file(target)
    if strategy and hasattr(strategy, "get_info"):
        info = strategy.get_info()
        metrics["strategy_name"] = info.get("name", target.stem)
        metrics["version"] = int(info.get("version", 1))
    else:
        metrics["strategy_name"] = target.stem
        try:
            metrics["version"] = int(target.stem.replace("strategy_v", ""))
        except ValueError:
            metrics["version"] = 1

    # 保存结果
    result_path = PROJECT_ROOT / "results" / f"{target.stem}.json"
    save_metrics_json(metrics, str(result_path))

    # 显示报告
    print(format_metrics_report(metrics, metrics["strategy_name"]))

    return metrics


# ============================================================
# Step 3b: 批量回测所有策略
# ============================================================

def auto_backtest_all(cfg):
    print("\n" + "=" * 60)
    print("  📊 批量回测所有策略（自动模式）")
    print("=" * 60)

    from data.fetcher import DataFetcher
    fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))

    if not strategies:
        print("❌ 没有找到策略文件！")
        return

    all_metrics = []
    for sp in strategies:
        metrics = auto_backtest(cfg, strategy_name=sp.name, fetcher=fetcher)
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        print("\n" + "=" * 70)
        print("  📊 批量回测汇总")
        print("=" * 70)
        print(f"{'策略':30s} {'总收益':>8s} {'夏普':>6s} {'回撤':>8s} {'卡玛':>6s}")
        print("-" * 70)
        best_sharpe = -999
        best_name = ""
        for m in all_metrics:
            name = m.get("strategy_name", "")[:30]
            tr = m.get("total_return", 0) * 100
            sr = m.get("sharpe_ratio", 0)
            dd = m.get("max_drawdown", 0) * 100
            cr = m.get("calmar_ratio", 0)
            print(f"{name:30s} {tr:7.2f}% {sr:6.2f} {dd:7.2f}% {cr:6.2f}")
            if sr > best_sharpe:
                best_sharpe = sr
                best_name = name
        print("-" * 70)
        print(f"\n🏆 最佳策略: {best_name} (夏普: {best_sharpe:.2f})")


# ============================================================
# Step 4: 绩效报告（无交互）
# ============================================================

def auto_report(cfg):
    print("\n" + "=" * 60)
    print("  📈 Step 4: 绩效报告（自动模式）")
    print("=" * 60)

    from engine.analyzer import format_metrics_report

    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))

    if not results:
        print("❌ 没有回测结果！请先运行 --step backtest")
        return

    print("\n📊 策略进化对比:")
    print("-" * 80)
    print(f"{'策略':25s} {'总收益':>8s} {'年化':>8s} {'回撤':>8s} {'夏普':>6s} {'卡玛':>6s} {'IR':>6s}")
    print("-" * 80)

    for r in results:
        with open(r) as f:
            m = json.load(f)
        name = m.get("strategy_name", r.stem)[:25]
        tr = m.get("total_return", 0) * 100
        ar = m.get("annual_return", 0) * 100
        dd = m.get("max_drawdown", 0) * 100
        sr = m.get("sharpe_ratio", 0)
        cr = m.get("calmar_ratio", 0)
        ir = m.get("information_ratio", 0)
        print(f"{name:25s} {tr:7.2f}% {ar:7.2f}% {dd:7.2f}% {sr:6.2f} {cr:6.2f} {ir:6.2f}")

    print("-" * 80)

    # 显示最新策略详细报告
    with open(results[-1]) as f:
        latest = json.load(f)
    name = latest.get("strategy_name", results[-1].stem)
    print(format_metrics_report(latest, name))


# ============================================================
# Step 5: AI 改进策略（无交互）
# ============================================================

def auto_evolve(cfg):
    print("\n" + "=" * 60)
    print("  🔄 Step 5: AI 改进策略（自动模式）")
    print("=" * 60)

    from llm.deepseek_client import create_client_from_config
    from engine.analyzer import metrics_to_feedback_context

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n❌ {e}")
        return None

    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))
    if not results:
        print("❌ 没有回测结果！请先运行 --step backtest")
        return None

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    if not strategies:
        print("❌ 没有策略文件！")
        return None

    with open(results[-1]) as f:
        metrics = json.load(f)

    strategy_name = metrics.get("strategy_name", "Unknown")
    strategy_version = metrics.get("version", len(results))
    latest_code = strategies[-1].read_text(encoding="utf-8")

    feedback = metrics_to_feedback_context(metrics, strategy_name, strategy_version)

    print(f"\n📊 当前策略: {strategy_name} (v{strategy_version})")
    print(f"   总收益: {metrics.get('total_return', 0)*100:.2f}%")
    print(f"   夏普: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"\n🤖 正在将结果发送给 DeepSeek 进行分析和改进...")

    code, error = client.analyze_and_improve(
        feedback_context=feedback,
        current_code=latest_code,
    )

    if error:
        print(f"\n❌ 策略改进失败: {error}")
        return None

    if not code:
        print("\n❌ 策略改进返回空内容")
        return None

    existing = list(pool_dir.glob("strategy_v*.py"))
    versions = [int(f.stem.replace("strategy_v", "")) for f in existing
                if f.stem.replace("strategy_v", "").isdigit()]
    version = max(versions) + 1 if versions else 1
    strategy_path = pool_dir / f"strategy_v{version}.py"
    strategy_path.write_text(code, encoding="utf-8")

    print(f"\n✅ 改进策略已保存: strategy_pool/strategy_v{version}.py")
    return str(strategy_path)


# ============================================================
# Step 6: 进化历史（无交互）
# ============================================================

def auto_history(cfg):
    print("\n" + "=" * 60)
    print("  📋 策略进化历史（自动模式）")
    print("=" * 60)

    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))

    print(f"\n策略文件: {len(strategies)} 个")
    print(f"回测结果: {len(results)} 个")

    if not results:
        print("暂无记录。")
        return

    print("\n" + "-" * 80)
    print(f"{'版本':6s} {'总收益':>8s} {'年化':>8s} {'回撤':>8s} {'夏普':>6s} {'卡玛':>6s} {'IR':>6s} {'胜率':>6s}")
    print("-" * 80)

    for r in results:
        with open(r) as f:
            m = json.load(f)
        ver = f"v{m.get('version', '?')}"
        tr = m.get("total_return", 0) * 100
        ar = m.get("annual_return", 0) * 100
        dd = m.get("max_drawdown", 0) * 100
        sr = m.get("sharpe_ratio", 0)
        cr = m.get("calmar_ratio", 0)
        ir = m.get("information_ratio", 0)
        wr = m.get("win_rate", 0) * 100
        print(f"{ver:6s} {tr:7.2f}% {ar:7.2f}% {dd:7.2f}% {sr:6.2f} {cr:6.2f} {ir:6.2f} {wr:5.1f}%")

    print("-" * 80)

    def _load(path):
        with open(path) as f:
            return json.load(f)

    best = max(results, key=lambda r: _load(r).get("sharpe_ratio", -999))
    best_m = _load(best)
    print(f"\n🏆 最佳策略: {best_m.get('strategy_name', best.stem)} (夏普: {best_m.get('sharpe_ratio', 0):.2f})")


# ============================================================
# 完整进化循环（无交互）
# ============================================================

def auto_full_cycle(cfg):
    """完整进化循环：下载数据 → 生成策略 → 回测 → 改进"""
    print("\n" + "=" * 60)
    print("  🧬 完整进化循环（自动模式）")
    print("=" * 60)

    # Step 1: 下载数据
    fetcher = auto_fetch(cfg)
    if fetcher is None:
        print("❌ 数据下载失败，终止")
        return

    # Step 2: 生成策略
    strategy_path = auto_generate(cfg)
    if strategy_path is None:
        print("❌ 策略生成失败，终止")
        return

    # Step 3: 回测
    metrics = auto_backtest(cfg, fetcher=fetcher)
    if metrics is None:
        print("❌ 回测失败，终止")
        return

    # Step 5: AI 改进
    auto_evolve(cfg)

    # 显示历史
    auto_history(cfg)


def auto_evolve_loop(cfg, n: int = 3):
    """连续进化 N 轮：每轮 = 回测最新策略 + AI 改进"""
    print("\n" + "=" * 60)
    print(f"  🔁 连续进化 {n} 轮（自动模式）")
    print("=" * 60)

    from data.fetcher import DataFetcher
    fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)

    for i in range(n):
        print(f"\n{'='*60}")
        print(f"  🔄 第 {i+1}/{n} 轮进化")
        print(f"{'='*60}")

        # 回测最新策略
        metrics = auto_backtest(cfg, fetcher=fetcher)
        if metrics is None:
            print(f"❌ 第 {i+1} 轮回测失败，跳过")
            continue

        sharpe = metrics.get("sharpe_ratio", 0)
        print(f"\n📊 本轮夏普比率: {sharpe:.2f}")

        if sharpe > 1.5:
            print(f"🎉 夏普比率已达到 {sharpe:.2f}，超过目标 1.5，停止进化！")
            break

        # AI 改进
        new_path = auto_evolve(cfg)
        if new_path is None:
            print(f"❌ 第 {i+1} 轮 AI 改进失败，跳过")

    # 最终报告
    auto_history(cfg)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quant Evolution 非交互式自动化运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 run_auto.py --step fetch
  python3 run_auto.py --step generate --instruction "生成一个动量反转策略"
  python3 run_auto.py --step backtest
  python3 run_auto.py --step backtest --strategy strategy_v8.py
  python3 run_auto.py --step backtest_all
  python3 run_auto.py --step report
  python3 run_auto.py --step evolve
  python3 run_auto.py --step history
  python3 run_auto.py --step full_cycle
  python3 run_auto.py --step evolve_loop --n 5
        """
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["fetch", "generate", "backtest", "backtest_all", "report",
                 "evolve", "history", "full_cycle", "evolve_loop"],
        help="要执行的步骤"
    )
    parser.add_argument("--strategy", type=str, default="",
                        help="指定回测的策略文件名，例如 strategy_v8.py")
    parser.add_argument("--instruction", type=str, default="",
                        help="策略生成指令（用于 generate 步骤）")
    parser.add_argument("--n", type=int, default=3,
                        help="进化轮数（用于 evolve_loop 步骤）")

    args = parser.parse_args()

    print(f"\n🚀 Quant Evolution 自动化运行")
    print(f"   项目路径: {PROJECT_ROOT}")
    print(f"   执行步骤: {args.step}")

    # 确保目录存在
    for d in ["data/cache", "strategy_pool", "results"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    try:
        cfg = load_config()
        print(f"   数据区间: {cfg.DATA_START_DATE} → {cfg.DATA_END_DATE}")
        print(f"   股票池: {cfg.STOCK_POOL}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)

    # 路由到对应步骤
    if args.step == "fetch":
        auto_fetch(cfg)
    elif args.step == "generate":
        auto_generate(cfg, instruction=args.instruction)
    elif args.step == "backtest":
        auto_backtest(cfg, strategy_name=args.strategy)
    elif args.step == "backtest_all":
        auto_backtest_all(cfg)
    elif args.step == "report":
        auto_report(cfg)
    elif args.step == "evolve":
        auto_evolve(cfg)
    elif args.step == "history":
        auto_history(cfg)
    elif args.step == "full_cycle":
        auto_full_cycle(cfg)
    elif args.step == "evolve_loop":
        auto_evolve_loop(cfg, n=args.n)


if __name__ == "__main__":
    main()
