#!/usr/bin/env python3
"""
============================================================
  🧬 Quant Evolution — 量化 AI 进化系统
  阶段 1: 人工闭环 — 交互式主控程序
============================================================

进化循环:
  1. 📊 下载 A股数据
  2. 🧠 DeepSeek 生成初始策略
  3. 🔬 运行回测
  4. 📈 查看绩效报告
  5. 🔄 反馈改进 → 生成下一版策略
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    """加载全局配置"""
    try:
        import config
        return config
    except ImportError:
        print("⚠️  config.py 未找到，使用默认配置")
        # 动态加载 config.example.py（文件名含点号，无法直接 import）
        import importlib.util
        example_path = PROJECT_ROOT / "config.example.py"
        if example_path.exists():
            spec = importlib.util.spec_from_file_location(
                "config_example", str(example_path)
            )
            config_example = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_example)
            return config_example
        else:
            raise FileNotFoundError("config.example.py 未找到！")


# ============================================================
# 菜单操作函数
# ============================================================

def step_fetch_data(cfg) -> Optional[object]:
    """
    Step 1: 下载/更新 A股数据
    """
    print("\n" + "=" * 52)
    print("  � Step 1: 下载/更新 A股数据")
    print("=" * 52)

    from data.fetcher import DataFetcher

    start = cfg.DATA_START_DATE
    end = cfg.DATA_END_DATE
    pool = cfg.STOCK_POOL

    print(f"  股票池: {pool}")
    print(f"  时间区间: {start} → {end}")
    print()

    try:
        fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)
        df = fetcher.fetch_daily(
            stock_pool=pool,
            start_date=start,
            end_date=end,
        )
        print(f"\n✅ 数据就绪: {len(df)} 行, {df['code'].nunique()} 支股票")
        return fetcher
    except Exception as e:
        print(f"\n❌ 数据获取失败: {e}")
        return None


def step_generate_strategy(cfg) -> Optional[str]:
    """
    Step 2: 用 DeepSeek 生成初始策略
    """
    print("\n" + "=" * 52)
    print("  🧠 Step 2: DeepSeek 生成策略")
    print("=" * 52)

    from llm.deepseek_client import create_client_from_config

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n❌ {e}")
        return None

    print("\n请描述你想要的策略方向（直接回车使用默认方向）:")
    print("  示例: '生成一个结合动量和低波动的选股策略'")
    print("  默认: '生成一个 A 股沪深300多因子选股策略'")
    instruction = input("\n📝 > ").strip()

    if not instruction:
        instruction = (
            "生成一个 A 股沪深300多因子选股策略。"
            "包含以下因子：20日动量、波动率过滤、换手率因子。"
            "月度调仓，持仓 20 支，等权重。"
        )

    print(f"\n🤖 正在调用 DeepSeek 生成策略...")
    print(f"   指令: {instruction[:80]}...")

    code, error = client.generate_strategy(instruction=instruction)

    if error:
        print(f"\n❌ 策略生成失败: {error}")
        return None

    if not code:
        print("\n❌ 策略生成返回空内容")
        return None

    # 确定版本号
    version = _get_next_version()

    # 保存策略文件
    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    strategy_path.write_text(code, encoding="utf-8")

    print(f"\n✅ 策略已保存: strategy_pool/strategy_v{version}.py")
    print(f"\n--- 策略代码预览 (前 30 行) ---")
    lines = code.split("\n")[:30]
    for line in lines:
        print(f"  {line}")

    return str(strategy_path)


def step_run_backtest(cfg, fetcher=None) -> Optional[Dict]:
    """
    Step 3: 运行回测
    """
    print("\n" + "=" * 52)
    print("  🔬 Step 3: 运行回测")
    print("=" * 52)

    # 列出可用策略
    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))

    if not strategies:
        print("\n❌ 没有找到策略文件！请先运行 Step 2 生成策略。")
        return None

    print("\n可用策略:")
    for i, s in enumerate(strategies):
        size = s.stat().st_size
        print(f"  [{i+1}] {s.name} ({size:,} bytes)")

    print(f"  [A] 运行最新策略")
    print(f"  [B] 运行所有策略")

    choice = input("\n📝 请选择 > ").strip().upper()

    if choice == "B":
        return _run_all_backtests(cfg, fetcher, strategies)
    else:
        if choice == "A" or not choice:
            target = strategies[-1]
        else:
            try:
                idx = int(choice) - 1
                target = strategies[idx]
            except (ValueError, IndexError):
                print("❌ 无效选择")
                return None

        return _run_single_backtest(cfg, fetcher, target)


def step_show_report(cfg) -> None:
    """
    Step 4: 查看绩效报告
    """
    print("\n" + "=" * 52)
    print("  📈 Step 4: 绩效报告")
    print("=" * 52)

    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))

    if not results:
        print("\n❌ 没有回测结果！请先运行 Step 3 进行回测。")
        return

    print("\n已有回测结果:")
    for i, r in enumerate(results):
        with open(r) as f:
            data = json.load(f)
        name = data.get("strategy_name", r.stem)
        total_ret = data.get("total_return", 0) * 100
        sharpe = data.get("sharpe_ratio", 0)
        print(f"  [{i+1}] {name} | 总收益: {total_ret:+.2f}% | 夏普: {sharpe:.2f}")

    print("\n选择要查看的详细报告 (直接回车看最新):")
    choice = input("📝 > ").strip()

    if choice:
        try:
            idx = int(choice) - 1
            target = results[idx]
        except (ValueError, IndexError):
            print("❌ 无效选择")
            return
    else:
        target = results[-1]

    # 显示报告
    with open(target) as f:
        metrics = json.load(f)

    from engine.analyzer import format_metrics_report
    name = metrics.get("strategy_name", target.stem)
    report = format_metrics_report(metrics, name)
    print(report)

    # 显示策略对比表
    if len(results) >= 2:
        print("\n📊 策略进化对比:")
        print("-" * 70)
        print(f"{'策略':20s} {'总收益':>8s} {'年化':>8s} {'回撤':>8s} {'夏普':>6s} {'卡玛':>6s}")
        print("-" * 70)
        for r in results:
            with open(r) as f:
                m = json.load(f)
            name = m.get("strategy_name", r.stem)[:20]
            tr = m.get("total_return", 0) * 100
            ar = m.get("annual_return", 0) * 100
            dd = m.get("max_drawdown", 0) * 100
            sr = m.get("sharpe_ratio", 0)
            cr = m.get("calmar_ratio", 0)
            print(f"{name:20s} {tr:7.2f}% {ar:7.2f}% {dd:7.2f}% {sr:6.2f} {cr:6.2f}")
        print("-" * 70)


def step_evolve(cfg) -> Optional[str]:
    """
    Step 5: 反馈改进 — 将回测结果发给 AI 生成改进版
    """
    print("\n" + "=" * 52)
    print("  🔄 Step 5: 反馈改进 — 策略进化")
    print("=" * 52)

    from llm.deepseek_client import create_client_from_config
    from engine.analyzer import metrics_to_feedback_context

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n❌ {e}")
        return None

    # 找到最新结果和最新策略
    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))
    if not results:
        print("\n❌ 没有回测结果！请先运行 Step 3。")
        return None

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    if not strategies:
        print("\n❌ 没有策略文件！")
        return None

    # 加载最新结果
    with open(results[-1]) as f:
        metrics = json.load(f)

    strategy_name = metrics.get("strategy_name", "Unknown")
    strategy_version = metrics.get("version", len(results))

    # 加载最新策略代码
    latest_code = strategies[-1].read_text(encoding="utf-8")

    # 生成反馈上下文
    feedback = metrics_to_feedback_context(
        metrics, strategy_name, strategy_version
    )

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

    # 保存新版本
    version = _get_next_version()
    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    strategy_path.write_text(code, encoding="utf-8")

    print(f"\n✅ 改进策略已保存: strategy_pool/strategy_v{version}.py")
    print(f"\n💡 建议: 立即运行 Step 3 回测，验证改进效果！")

    return str(strategy_path)


def step_show_history(cfg) -> None:
    """
    Step 6: 查看进化历史
    """
    print("\n" + "=" * 52)
    print("  📋 策略进化历史")
    print("=" * 52)

    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))

    print(f"\n策略文件: {len(strategies)} 个")
    print(f"回测结果: {len(results)} 个")
    print()

    if not results:
        print("暂无记录。")
        return

    print("-" * 80)
    print(f"{'版本':6s} {'总收益':>8s} {'年化':>8s} {'回撤':>8s} {'夏普':>6s} {'卡玛':>6s} {'IR':>6s}")
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
        print(f"{ver:6s} {tr:7.2f}% {ar:7.2f}% {dd:7.2f}% {sr:6.2f} {cr:6.2f} {ir:6.2f}")

    print("-" * 80)

    # 找最佳策略
    best = max(results, key=lambda r: json.load(open(r)).get("sharpe_ratio", -999))
    with open(best) as f:
        best_m = json.load(f)
    print(f"\n🏆 最佳策略: {best_m.get('strategy_name', best.stem)} (夏普: {best_m.get('sharpe_ratio', 0):.2f})")


def step_fix_code(cfg) -> Optional[str]:
    """
    Step 7 (隐藏): 手动修复最近策略的报错
    """
    print("\n" + "=" * 52)
    print("  🔧 Step 7: 代码修复")
    print("=" * 52)

    from llm.deepseek_client import create_client_from_config

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n❌ {e}")
        return None

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    if not strategies:
        print("\n❌ 没有策略文件！")
        return None

    latest = strategies[-1]
    code = latest.read_text(encoding="utf-8")

    print(f"\n当前策略: {latest.name}")
    print("\n请输入报错信息 (可多行，输入空行结束):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    error_msg = "\n".join(lines)
    if not error_msg.strip():
        print("❌ 未输入报错信息")
        return None

    print(f"\n🤖 正在请求 DeepSeek 修复...")

    fixed_code, error = client.fix_code(
        error_message=error_msg,
        strategy_code=code,
    )

    if error:
        print(f"\n❌ 修复失败: {error}")
        return None

    if not fixed_code:
        print("\n❌ 修复返回空内容")
        return None

    version = _get_next_version()
    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    strategy_path.write_text(fixed_code, encoding="utf-8")

    print(f"\n✅ 修复代码已保存: strategy_pool/strategy_v{version}.py")
    return str(strategy_path)


# ============================================================
# 辅助函数
# ============================================================

def _get_next_version() -> int:
    """获取下一个策略版本号"""
    pool_dir = PROJECT_ROOT / "strategy_pool"
    existing = list(pool_dir.glob("strategy_v*.py"))
    if not existing:
        return 1

    versions = []
    for f in existing:
        try:
            v = int(f.stem.replace("strategy_v", ""))
            versions.append(v)
        except ValueError:
            continue

    return max(versions) + 1 if versions else 1


def _run_single_backtest(cfg, fetcher, strategy_path) -> Optional[Dict]:
    """运行单个策略回测"""
    import pandas as pd
    from data.fetcher import DataFetcher
    from engine.sandbox import execute_strategy, format_error_for_llm
    from engine.backtest import run_backtest_simple
    from engine.analyzer import compute_metrics, save_metrics_json, format_metrics_report
    from llm.deepseek_client import create_client_from_config

    print(f"\n🔬 回测: {strategy_path.name}")

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
    signals, error = execute_strategy(strategy_path, data)

    if error:
        print(f"\n❌ 策略执行失败:")
        print(f"   {error}")

        # 自动尝试修复
        fix = input("\n🔧 是否让 AI 自动修复？[Y/n]: ").strip().lower()
        if fix in ("", "y", "yes"):
            code = strategy_path.read_text(encoding="utf-8")
            fix_prompt = format_error_for_llm(error, code)

            try:
                client = create_client_from_config()
                fixed_code, fix_error = client.fix_code(
                    error_message=error,
                    strategy_code=code,
                )

                if fixed_code and not fix_error:
                    version = _get_next_version()
                    new_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
                    new_path.write_text(fixed_code, encoding="utf-8")
                    print(f"   ✅ 修复代码已保存: {new_path.name}")
                    print(f"   💡 请重新运行 Step 3 测试修复后的策略。")
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
    from engine.sandbox import load_strategy_from_file
    strategy, _ = load_strategy_from_file(strategy_path)
    if strategy and hasattr(strategy, "get_info"):
        info = strategy.get_info()
        metrics["strategy_name"] = info.get("name", strategy_path.stem)
        metrics["version"] = int(info.get("version", 1))
    else:
        metrics["strategy_name"] = strategy_path.stem
        metrics["version"] = _get_version_from_path(strategy_path)

    # 保存结果
    result_path = PROJECT_ROOT / "results" / f"{strategy_path.stem}.json"
    save_metrics_json(metrics, str(result_path))

    # 显示报告
    print(format_metrics_report(metrics, metrics["strategy_name"]))

    return metrics


def _run_all_backtests(cfg, fetcher, strategies) -> Optional[Dict]:
    """批量运行所有策略"""
    print(f"\n📊 批量回测 {len(strategies)} 个策略...")

    all_metrics = []
    for sp in strategies:
        metrics = _run_single_backtest(cfg, fetcher, sp)
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        print("\n" + "=" * 70)
        print("  📊 批量回测汇总")
        print("=" * 70)
        print(f"{'策略':30s} {'总收益':>8s} {'夏普':>6s} {'回撤':>8s}")
        print("-" * 70)
        best_sharpe = -999
        best_name = ""
        for m in all_metrics:
            name = m.get("strategy_name", "")[:30]
            tr = m.get("total_return", 0) * 100
            sr = m.get("sharpe_ratio", 0)
            dd = m.get("max_drawdown", 0) * 100
            print(f"{name:30s} {tr:7.2f}% {sr:6.2f} {dd:7.2f}%")
            if sr > best_sharpe:
                best_sharpe = sr
                best_name = name
        print("-" * 70)
        print(f"\n🏆 最佳策略: {best_name} (夏普: {best_sharpe:.2f})")

    return all_metrics[-1] if all_metrics else None


def _get_version_from_path(path) -> int:
    """从文件路径提取版本号"""
    try:
        return int(path.stem.replace("strategy_v", ""))
    except ValueError:
        return 1


# ============================================================
# 主菜单
# ============================================================

def show_menu():
    """显示主菜单"""
    print("""
╔══════════════════════════════════════════════════╗
║         🧬 Quant Evolution — Phase 1             ║
║           量化 AI 进化系统 · 人工闭环             ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  1. 📊 下载/更新 A股数据                          ║
║  2. 🧠 用 DeepSeek 生成初始策略                   ║
║  3. 🔬 运行回测                                   ║
║  4. 📈 查看绩效报告                               ║
║  5. 🔄 将回测结果反馈给 AI 改进策略                ║
║  6. 📋 查看策略进化历史                           ║
║  7. 🔧 手动修复策略报错                           ║
║  0. 👋 退出                                       ║
║                                                  ║
╚══════════════════════════════════════════════════╝
""")


def main():
    """主入口"""
    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)

    print(f"\n🚀 启动 Quant Evolution 系统...")
    print(f"   项目路径: {PROJECT_ROOT}")

    # 加载配置
    try:
        cfg = load_config()
        print(f"   数据区间: {cfg.DATA_START_DATE} → {cfg.DATA_END_DATE}")
        print(f"   股票池: {cfg.STOCK_POOL}")
        print(f"   初始资金: {cfg.INITIAL_CAPITAL:,.0f} 元")
    except Exception as e:
        print(f"   ⚠️ 配置加载警告: {e}")
        return

    # 状态变量
    fetcher = None

    # 主循环
    while True:
        show_menu()
        choice = input("📝 请选择 [0-7] > ").strip()

        if choice == "0":
            print("\n👋 再见！记住：量化之路，贵在坚持。")
            break
        elif choice == "1":
            fetcher = step_fetch_data(cfg)
        elif choice == "2":
            step_generate_strategy(cfg)
        elif choice == "3":
            step_run_backtest(cfg, fetcher)
        elif choice == "4":
            step_show_report(cfg)
        elif choice == "5":
            step_evolve(cfg)
        elif choice == "6":
            step_show_history(cfg)
        elif choice == "7":
            step_fix_code(cfg)
        else:
            print("❌ 无效选择，请输入 0-7")

        # 每步操作后暂停
        if choice != "0":
            input("\n⏎ 按回车键继续...")


if __name__ == "__main__":
    main()