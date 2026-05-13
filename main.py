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

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    """加载全局配置"""
    try:
        import config
        return config
    except ImportError:
        print("⚠️  config.py 未找到，使用默认配置")
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


def ensure_directories() -> None:
    """Ensure core project directories exist."""
    for path in [PROJECT_ROOT / "data" / "cache", PROJECT_ROOT / "strategy_pool", PROJECT_ROOT / "results"]:
        path.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Quant Evolution 量化AI回测系统")
    parser.add_argument("--step", type=int, choices=range(1, 10), help="Run specific step (1-9)")
    parser.add_argument("--strategy", type=str, help="Specify strategy file, e.g. strategy_v1.py")
    parser.add_argument("--all", action="store_true", help="Run all steps 1-6 sequentially")
    return parser.parse_args()


# ============================================================
# 菜单操作函数
# ============================================================

def step_fetch_data(cfg) -> Optional[object]:
    """Step 1: Download/Update A-share data"""
    print("\n" + "=" * 52)
    print("  1. Download/Update A-share data")
    print("=" * 52)

    from data.fetcher import DataFetcher

    start = cfg.DATA_START_DATE
    end = cfg.DATA_END_DATE
    pool = cfg.STOCK_POOL

    print(f"  Stock pool: {pool}")
    print(f"  Date range: {start} -> {end}")
    print()

    try:
        fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)
        df = fetcher.fetch_daily(stock_pool=pool, start_date=start, end_date=end)
        print(f"\nData ready: {len(df)} rows, {df['code'].nunique()} stocks")
        return fetcher
    except Exception as e:
        print(f"\nData fetch failed: {e}")
        return None


def step_generate_strategy(cfg) -> Optional[str]:
    """Step 2: DeepSeek generates initial strategy"""
    print("\n" + "=" * 52)
    print("  2. DeepSeek Strategy Generation")
    print("=" * 52)

    from llm.deepseek_client import create_client_from_config

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n{e}")
        return None

    print("\nDescribe your strategy direction (Enter for default):")
    print("  Example: 'Generate a strategy combining momentum and low volatility'")
    print("  Default: 'Generate a CSI300 multi-factor A-share strategy'")
    instruction = input("\n> ").strip()

    if not instruction:
        instruction = (
            "Generate a CSI300 multi-factor A-share stock selection strategy. "
            "Factors: 20-day momentum, volatility filter, turnover rate factor. "
            "Monthly rebalance, 20 holdings, equal weight."
        )

    print(f"\nCalling DeepSeek to generate strategy...")
    print(f"   Instruction: {instruction[:80]}...")

    code, error = client.generate_strategy(instruction=instruction)

    if error:
        print(f"\nStrategy generation failed: {error}")
        return None

    if not code:
        print("\nStrategy generation returned empty content")
        return None

    version = _get_next_version()
    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    strategy_path.write_text(code, encoding="utf-8")

    print(f"\nStrategy saved: strategy_pool/strategy_v{version}.py")
    print(f"\n--- Code preview (first 30 lines) ---")
    lines = code.split("\n")[:30]
    for line in lines:
        print(f"  {line}")

    return str(strategy_path)


def step_run_backtest(cfg, fetcher=None, strategy_choice: Optional[str] = None) -> Optional[Dict]:
    """Step 3: Run backtest"""
    print("\n" + "=" * 52)
    print("  3. Run Backtest")
    print("=" * 52)

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))

    if not strategies:
        print("\nNo strategy files found! Run Step 2 first.")
        return None

    print("\nAvailable strategies:")
    for i, s in enumerate(strategies):
        size = s.stat().st_size
        print(f"  [{i+1}] {s.name} ({size:,} bytes)")

    print(f"  [A] Run latest strategy")
    print(f"  [B] Run ALL strategies")

    choice = strategy_choice
    if choice is None:
        choice = input("\nSelect > ").strip().upper()
    else:
        choice = choice.strip().upper()
        print(f"\nSelect: {choice}")

    if choice == "B":
        return _run_all_backtests(cfg, fetcher, strategies)

    if choice == "A" or not choice:
        target = strategies[-1]
    else:
        try:
            idx = int(choice) - 1
            target = strategies[idx]
        except (ValueError, IndexError):
            print("Invalid selection")
            return None

    return _run_single_backtest(cfg, fetcher, target)


def step_show_report(cfg) -> None:
    """Step 4: View performance report"""
    print("\n" + "=" * 52)
    print("  4. Performance Report")
    print("=" * 52)

    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))

    if not results:
        print("\nNo backtest results! Run Step 3 first.")
        return

    print("\nExisting backtest results:")
    for i, r in enumerate(results):
        with open(r) as f:
            data = json.load(f)
        name = data.get("strategy_name", r.stem)
        total_ret = data.get("total_return", 0) * 100
        sharpe = data.get("sharpe_ratio", 0)
        print(f"  [{i+1}] {name} | Total Return: {total_ret:+.2f}% | Sharpe: {sharpe:.2f}")

    print("\nSelect report to view (Enter for latest):")
    choice = input("> ").strip()

    if choice:
        try:
            idx = int(choice) - 1
            target = results[idx]
        except (ValueError, IndexError):
            print("Invalid selection")
            return
    else:
        target = results[-1]

    with open(target) as f:
        metrics = json.load(f)

    from engine.analyzer import format_metrics_report
    name = metrics.get("strategy_name", target.stem)
    report = format_metrics_report(metrics, name)
    print(report)

    if len(results) >= 2:
        print("\nStrategy Evolution Comparison:")
        print("-" * 70)
        print(f"{'Strategy':20s} {'TotRet':>8s} {'AnnRet':>8s} {'MaxDD':>8s} {'Sharpe':>6s} {'Calmar':>6s}")
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
    """Step 5: Feedback & Evolution — select a strategy and let AI improve it"""
    print("\n" + "=" * 52)
    print("  5. Feedback & Evolution")
    print("=" * 52)

    from llm.deepseek_client import create_client_from_config
    from engine.analyzer import metrics_to_feedback_context

    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))
    if not results:
        print("\nNo backtest results! Run Step 3 first.")
        return None

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    if not strategies:
        print("\nNo strategy files!")
        return None

    print("\nSelectable strategies to improve:")
    print("-" * 75)
    print(f"{'#':3s} {'Version':8s} {'Strategy Name':25s} {'TotRet':>8s} {'Sharpe':>7s}")
    print("-" * 75)
    for i, r in enumerate(results):
        with open(r) as f:
            m = json.load(f)
        ver = f"v{m.get('version', '?')}"
        name = m.get("strategy_name", r.stem)[:25]
        tr = m.get("total_return", 0) * 100
        sr = m.get("sharpe_ratio", 0)
        print(f"{i+1:3d} {ver:8s} {name:25s} {tr:7.2f}% {sr:7.2f}")
    print("-" * 75)
    print(f"  [A] Latest strategy (v{len(results)})")
    print(f"  [B] Best Sharpe ratio strategy")

    choice = input("\nSelect strategy to improve > ").strip().upper()

    selected_result = None
    if choice == "A" or choice == "":
        selected_result = results[-1]
    elif choice == "B":
        def _load_result(path):
            with open(path) as f:
                return json.load(f)
        selected_result = max(results, key=lambda r: _load_result(r).get("sharpe_ratio", -999))
    else:
        try:
            idx = int(choice) - 1
            selected_result = results[idx]
        except (ValueError, IndexError):
            print("Invalid selection")
            return None

    with open(selected_result) as f:
        metrics = json.load(f)

    strategy_name = metrics.get("strategy_name", "Unknown")
    strategy_version = metrics.get("version", len(results))

    strategy_path = pool_dir / selected_result.stem.replace(".json", ".py")
    if strategy_path.exists():
        current_code = strategy_path.read_text(encoding="utf-8")
    else:
        current_code = strategies[-1].read_text(encoding="utf-8")

    feedback = metrics_to_feedback_context(metrics, strategy_name, strategy_version)

    print(f"\nSelected strategy: {strategy_name} (v{strategy_version})")
    print(f"   Total Return: {metrics.get('total_return', 0)*100:.2f}%")
    print(f"   Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Max DD: {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"\nSending to DeepSeek for analysis and improvement...")

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n{e}")
        return None

    code, error = client.analyze_and_improve(feedback_context=feedback, current_code=current_code)

    if error:
        print(f"\nImprovement failed: {error}")
        return None

    if not code:
        print("\nImprovement returned empty content")
        return None

    version = _get_next_version()
    new_strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    new_strategy_path.write_text(code, encoding="utf-8")

    print(f"\nImproved strategy saved: strategy_pool/strategy_v{version}.py")
    print(f"\nTip: Run Step 3 now to verify improvements!")

    return str(new_strategy_path)


def step_show_history(cfg) -> None:
    """Step 6: View evolution history"""
    print("\n" + "=" * 52)
    print("  6. Evolution History")
    print("=" * 52)

    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))

    print(f"\nStrategy files: {len(strategies)}")
    print(f"Backtest results: {len(results)}")
    print()

    if not results:
        print("No records yet.")
        return

    print("-" * 80)
    print(f"{'Ver':6s} {'TotRet':>8s} {'AnnRet':>8s} {'MaxDD':>8s} {'Sharpe':>6s} {'Calmar':>6s} {'IR':>6s}")
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

    def _load_result(path):
        with open(path) as f:
            return json.load(f)
    best = max(results, key=lambda r: _load_result(r).get("sharpe_ratio", -999))
    best_m = _load_result(best)
    print(f"\nBest strategy: {best_m.get('strategy_name', best.stem)} (Sharpe: {best_m.get('sharpe_ratio', 0):.2f})")


def step_delete_strategy(cfg) -> None:
    """Step 8: Delete strategy files and/or backtest results"""
    print("\n" + "=" * 52)
    print("  8. Delete Strategy")
    print("=" * 52)

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))

    if not strategies and not results:
        print("\nNothing to delete.")
        return

    print("\nExisting strategies:")
    print("-" * 75)
    print(f"{'#':3s} {'Version':10s} {'Strategy File':25s} {'Has Result':10s} {'File Size':>8s}")
    print("-" * 75)

    result_stems = {r.stem for r in results}
    for i, s in enumerate(strategies):
        has_result = "Yes" if s.stem in result_stems else "No"
        size = f"{s.stat().st_size:,} B"
        print(f"{i+1:3d} {s.stem:10s} {s.name:25s} {has_result:10s} {size:>8s}")

    print("-" * 75)
    print(f"  [A] Delete ALL strategies AND results")
    print(f"  [B] Delete latest strategy")
    print(f"  [C] Delete ALL backtest results only (keep strategy code)")
    print(f"  [0] Cancel")

    choice = input("\nSelect > ").strip().upper()

    if choice == "0" or choice == "":
        print("  Cancelled.")
        return

    if choice == "A":
        confirm = input("Confirm delete ALL strategy files and results? [yes/NO]: ").strip()
        if confirm != "yes":
            print("  Cancelled.")
            return
        for s in strategies:
            s.unlink()
        for r in results:
            r.unlink()
        print(f"Deleted {len(strategies)} strategy files, {len(results)} backtest results.")
        return

    if choice == "B":
        target_st = strategies[-1]
    elif choice == "C":
        for r in results:
            r.unlink()
        print(f"Deleted {len(results)} backtest results (strategy files preserved).")
        return
    else:
        try:
            idx = int(choice) - 1
            target_st = strategies[idx]
        except (ValueError, IndexError):
            print("Invalid selection")
            return

    target_name = target_st.stem
    confirm = input(f"Confirm delete {target_name}? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("  Cancelled.")
        return

    target_st.unlink()
    print(f"Deleted strategy: {target_st.name}")

    result_path = results_dir / f"{target_name}.json"
    if result_path.exists():
        result_path.unlink()
        print(f"Deleted result: {result_path.name}")


def step_chat_with_ai(cfg) -> None:
    """
    Step 9: Chat with AI — 自由对话模式

    提供上下文：
    - 当前策略池概览
    - 回测结果摘要
    - 用户可以自由提问（因子设计、市场分析、代码改进等）
    """
    print("\n" + "=" * 52)
    print("  9. AI Chat")
    print("=" * 52)

    from llm.deepseek_client import create_client_from_config

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n{e}")
        return

    # ---- 构建系统上下文 ----
    context_parts = ["你是一个顶级的量化投资助手，当前系统是一个「量化AI进化系统」。"]

    # 策略池概览
    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    if strategies:
        context_parts.append(f"\n## 当前策略池\n共 {len(strategies)} 个策略文件：")
        for s in strategies:
            version = s.stem.replace("strategy_v", "")
            context_parts.append(f"- strategy_v{version}.py")
    else:
        context_parts.append("\n## 当前策略池\n（空）尚未生成任何策略。")

    # 回测结果摘要
    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))
    if results:
        context_parts.append(f"\n## 回测结果摘要\n共 {len(results)} 条记录：")
        for r in results:
            with open(r) as f:
                m = json.load(f)
            name = m.get("strategy_name", r.stem)
            tr = m.get("total_return", 0) * 100
            sr = m.get("sharpe_ratio", 0)
            dd = m.get("max_drawdown", 0) * 100
            context_parts.append(f"- {name}: 总收益 {tr:+.2f}%, 夏普 {sr:.2f}, 回撤 {dd:.2f}%")
    else:
        context_parts.append("\n## 回测结果\n（空）尚未运行回测。")

    context_parts.append("\n## 当前配置")
    context_parts.append(f"- 股票池: {cfg.STOCK_POOL}")
    context_parts.append(f"- 数据区间: {cfg.DATA_START_DATE} → {cfg.DATA_END_DATE}")
    context_parts.append(f"- 初始资金: {cfg.INITIAL_CAPITAL:,.0f} 元")
    context_parts.append(f"- 回测引擎: VectorBT")

    context_parts.append(
        "\n用户可能会就以下话题向你提问："
        "\n- 如何设计新的选股因子（A股场景）"
        "\n- 当前策略的改进方向"
        "\n- 风险管理和仓位优化"
        "\n- 市场行情分析"
        "\n- Python/量化编程问题"
    )

    system_prompt = "\n".join(context_parts)

    print("\n系统上下文已加载:")
    print(f"  - 策略池: {len(strategies)} 个文件")
    print(f"  - 回测结果: {len(results)} 条")
    print(f"  - 股票池: {cfg.STOCK_POOL}")
    print()
    print("🤖 AI 量化助手已就绪！输入你的问题开始对话。")
    print("   输入 'exit' 或 'quit' 退出对话。")
    print("   输入 '/code' 切换为代码生成模式（AI 只输出策略代码）。")
    print("-" * 60)

    # 对话历史
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    code_mode = False  # 普通对话模式

    while True:
        try:
            user_input = input("\n🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n   [退出对话]")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.lower() in ("exit", "quit", "退出"):
            print("   [退出 AI 对话]")
            break

        if user_input.lower() in ("/code", "/策略"):
            code_mode = not code_mode
            status = "✅ 代码模式已开启 (AI 将只输出 Python 策略代码)" if code_mode else "🔄 已切换回普通对话模式"
            print(f"   {status}")
            continue

        if user_input.lower() in ("/help", "/帮助"):
            print("\n   可用命令:")
            print("   /code    - 切换代码生成模式")
            print("   /context - 显示当前系统上下文")
            print("   exit     - 退出对话")
            continue

        if user_input.lower() == "/context":
            print(f"\n   📋 系统上下文 (前20行):")
            for line in system_prompt.split("\n")[:20]:
                print(f"   {line}")
            continue

        # 构建用户消息
        if code_mode:
            user_message = (
                f"{user_input}\n\n"
                "请直接输出完整的 Python 策略代码（继承 BaseStrategy），"
                "用 ```python 包裹，不要任何解释文字。"
            )
        else:
            user_message = user_input

        messages.append({"role": "user", "content": user_message})

        # 调用 AI
        print("   🤖 思考中...", end=" ", flush=True)
        response, error = client.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.8,
            max_tokens=2048,
        )

        if error:
            print(f"\n   ❌ {error}")
            messages.pop()  # 移除失败的消息
            continue

        if not response:
            print("\n   ❌ AI 返回空响应")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": response})

        # 显示回复
        print("\r" + "-" * 60)
        print(response)
        print("-" * 60)

        # 代码模式下可保存
        if code_mode and "```python" in response:
            save = input("\n   💾 是否保存此代码为新策略？[y/N]: ").strip().lower()
            if save in ("y", "yes"):
                import re
                code_match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    version = _get_next_version()
                    sp = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
                    sp.write_text(code, encoding="utf-8")
                    print(f"   ✅ 已保存: strategy_pool/strategy_v{version}.py")
                else:
                    print("   ⚠️ 未能提取代码块")


def step_fix_code(cfg) -> Optional[str]:
    """Step 7: Manually fix strategy errors via AI"""
    print("\n" + "=" * 52)
    print("  7. Code Fix (AI)")
    print("=" * 52)

    from llm.deepseek_client import create_client_from_config

    try:
        client = create_client_from_config()
    except ValueError as e:
        print(f"\n{e}")
        return None

    pool_dir = PROJECT_ROOT / "strategy_pool"
    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    if not strategies:
        print("\nNo strategy files!")
        return None

    latest = strategies[-1]
    code = latest.read_text(encoding="utf-8")

    print(f"\nCurrent strategy: {latest.name}")
    print("\nPaste error message (empty line to finish):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    error_msg = "\n".join(lines)
    if not error_msg.strip():
        print("No error message provided")
        return None

    print(f"\nRequesting DeepSeek to fix...")

    fixed_code, error = client.fix_code(error_message=error_msg, strategy_code=code)

    if error:
        print(f"\nFix failed: {error}")
        return None

    if not fixed_code:
        print("\nFix returned empty content")
        return None

    version = _get_next_version()
    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    strategy_path.write_text(fixed_code, encoding="utf-8")

    print(f"\nFixed code saved: strategy_pool/strategy_v{version}.py")
    return str(strategy_path)


# ============================================================
# 辅助函数
# ============================================================

def _get_next_version() -> int:
    """Get next strategy version number"""
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
    """Run single strategy backtest"""
    import pandas as pd
    from data.fetcher import DataFetcher
    from engine.sandbox import execute_strategy, format_error_for_llm
    from engine.backtest import run_backtest_simple
    from engine.analyzer import compute_metrics, save_metrics_json, format_metrics_report
    from llm.deepseek_client import create_client_from_config

    print(f"\nBacktesting: {strategy_path.name}")

    if fetcher is None:
        fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)

    data = fetcher.get_data(
        start_date=cfg.DATA_START_DATE,
        end_date=cfg.DATA_END_DATE,
        universe=cfg.STOCK_POOL,
    )

    print("   Executing strategy in sandbox...")
    signals, error = execute_strategy(strategy_path, data)

    if error:
        print(f"\nStrategy execution failed:")
        print(f"   {error}")

        fix = input("\nAuto-fix with AI? [Y/n]: ").strip().lower()
        if fix in ("", "y", "yes"):
            code = strategy_path.read_text(encoding="utf-8")
            try:
                client = create_client_from_config()
                fixed_code, fix_error = client.fix_code(error_message=error, strategy_code=code)
                if fixed_code and not fix_error:
                    version = _get_next_version()
                    new_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
                    new_path.write_text(fixed_code, encoding="utf-8")
                    print(f"   Fixed code saved: {new_path.name}")
                    print(f"   Tip: Re-run Step 3 to test the fixed strategy.")
                else:
                    print(f"   AI fix failed: {fix_error}")
            except Exception as e:
                print(f"   Cannot connect to LLM: {e}")
        return None

    print("   Running VectorBT backtest...")
    result = run_backtest_simple(
        data=data, signals=signals,
        initial_capital=cfg.INITIAL_CAPITAL,
        commission=cfg.COMMISSION,
        slippage=cfg.SLIPPAGE,
    )

    if "error" in result and result["error"]:
        print(f"   Backtest failed: {result['error']}")
        return None

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

    equity = result.get("equity_curve", pd.Series(dtype=float))
    trades = result.get("trades")

    metrics = compute_metrics(
        equity_curve=equity, trades=trades,
        risk_free_rate=cfg.RISK_FREE_RATE,
        benchmark_returns=benchmark_returns,
    )

    from engine.sandbox import load_strategy_from_file
    strategy, _ = load_strategy_from_file(strategy_path)
    if strategy and hasattr(strategy, "get_info"):
        info = strategy.get_info()
        metrics["strategy_name"] = info.get("name", strategy_path.stem)
        metrics["version"] = int(info.get("version", 1))
    else:
        metrics["strategy_name"] = strategy_path.stem
        metrics["version"] = _get_version_from_path(strategy_path)

    result_path = PROJECT_ROOT / "results" / f"{strategy_path.stem}.json"
    save_metrics_json(metrics, str(result_path))

    print(format_metrics_report(metrics, metrics["strategy_name"]))
    return metrics


def _run_all_backtests(cfg, fetcher, strategies) -> Optional[Dict]:
    """Batch run all strategies"""
    print(f"\nBatch backtesting {len(strategies)} strategies...")

    all_metrics = []
    for sp in strategies:
        metrics = _run_single_backtest(cfg, fetcher, sp)
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        print("\n" + "=" * 70)
        print("  Batch Summary")
        print("=" * 70)
        print(f"{'Strategy':30s} {'TotRet':>8s} {'Sharpe':>6s} {'MaxDD':>8s}")
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
        print(f"\nBest strategy: {best_name} (Sharpe: {best_sharpe:.2f})")

    return all_metrics[-1] if all_metrics else None


def _get_version_from_path(path) -> int:
    """Extract version number from file path"""
    try:
        return int(path.stem.replace("strategy_v", ""))
    except ValueError:
        return 1


# ============================================================
# 主菜单
# ============================================================

def show_menu():
    """Display main menu"""
    print("""
+====================================================+
|         Quant Evolution -- Phase 1                 |
|        AI Quantitative Evolution System            |
+====================================================+
|                                                    |
|  1. Download/Update A-share data                   |
|  2. DeepSeek generate initial strategy             |
|  3. Run backtest                                   |
|  4. View performance report                        |
|  5. Feedback results to AI for improvement         |
|  6. View evolution history                         |
|  7. Manually fix strategy errors (AI)              |
|  8. Delete strategies                              |
|  9. AI Chat                                        |
|  0. Exit                                           |
|                                                    |
+====================================================+
""")


def main():
    """Main entry"""
    os.chdir(PROJECT_ROOT)
    ensure_directories()

    args = parse_args()

    print(f"\nQuant Evolution System Starting...")
    print(f"   Project path: {PROJECT_ROOT}")

    try:
        cfg = load_config()
        print(f"   Date range: {cfg.DATA_START_DATE} -> {cfg.DATA_END_DATE}")
        print(f"   Stock pool: {cfg.STOCK_POOL}")
        print(f"   Initial capital: {cfg.INITIAL_CAPITAL:,.0f}")
    except Exception as e:
        print(f"   Config load warning: {e}")
        return

    fetcher = None

    # CLI mode
    if args.all:
        for step in range(1, 7):
            if step == 1:
                fetcher = step_fetch_data(cfg)
            elif step == 2:
                step_generate_strategy(cfg)
            elif step == 3:
                step_run_backtest(cfg, fetcher, strategy_choice=args.strategy)
            elif step == 4:
                step_show_report(cfg)
            elif step == 5:
                step_evolve(cfg)
            elif step == 6:
                step_show_history(cfg)
        return

    if args.step is not None:
        mapping = {1: step_fetch_data, 2: step_generate_strategy, 3: step_run_backtest,
                   4: step_show_report, 5: step_evolve, 6: step_show_history,
                   7: step_fix_code, 8: step_delete_strategy, 9: step_chat_with_ai}
        func = mapping.get(args.step)
        if func:
            if args.step == 3:
                func(cfg, fetcher, strategy_choice=args.strategy)
            else:
                func(cfg)
        return

    # Interactive mode
    while True:
        show_menu()
        choice = input("Select [0-9] > ").strip()

        if choice == "0":
            print("\nGoodbye!")
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
        elif choice == "8":
            step_delete_strategy(cfg)
        elif choice == "9":
            step_chat_with_ai(cfg)
        else:
            print("Invalid selection, please enter 0-9")

        if choice != "0":
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()