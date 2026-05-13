"""
Bridge — 封装现有 engine/data/llm 模块，为 Web API 提供统一调用接口
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# 确保能导入父目录模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    """加载全局配置"""
    try:
        import config
        return config
    except ImportError:
        import importlib.util
        example_path = PROJECT_ROOT / "config.example.py"
        spec = importlib.util.spec_from_file_location("config_example", str(example_path))
        config_example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_example)
        return config_example


def get_strategies() -> List[Dict]:
    """获取所有策略文件列表（含已有回测结果摘要）"""
    pool_dir = PROJECT_ROOT / "strategy_pool"
    results_dir = PROJECT_ROOT / "results"

    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    result_map = {}
    for r in sorted(results_dir.glob("strategy_v*.json")):
        try:
            with open(r) as f:
                m = json.load(f)
            result_map[r.stem] = {
                "total_return": m.get("total_return", 0),
                "annual_return": m.get("annual_return", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "calmar_ratio": m.get("calmar_ratio", 0),
                "win_rate": m.get("win_rate", 0),
                "total_trades": m.get("total_trades", 0),
                "strategy_name": m.get("strategy_name", r.stem),
                "has_result": True,
            }
        except Exception:
            pass

    result = []
    for s in strategies:
        version = _extract_version(s.stem)
        entry = {
            "version": version,
            "filename": s.name,
            "size": s.stat().st_size,
            "has_result": s.stem in result_map,
        }
        if s.stem in result_map:
            entry.update(result_map[s.stem])
        result.append(entry)

    # 只读结果但无对应策略文件的
    optional_stems = {s.stem for s in strategies}
    for stem, rm in result_map.items():
        if stem not in optional_stems:
            entry = {"version": _extract_version(stem), "filename": stem + ".py", "size": 0, "has_result": True}
            entry.update(rm)
            result.append(entry)

    result.sort(key=lambda x: x["version"])
    return result


def get_strategy_code(version: int) -> Tuple[Optional[str], Optional[str]]:
    """读取策略源代码"""
    path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    if not path.exists():
        return None, f"策略 v{version} 不存在"
    return path.read_text(encoding="utf-8"), None


def delete_strategy(version: int) -> Tuple[bool, str]:
    """删除策略文件及其回测结果"""
    py_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    json_path = PROJECT_ROOT / "results" / f"strategy_v{version}.json"

    deleted = []
    if py_path.exists():
        py_path.unlink()
        deleted.append(f"strategy_v{version}.py")
    if json_path.exists():
        json_path.unlink()
        deleted.append(f"strategy_v{version}.json")

    if deleted:
        return True, f"已删除: {', '.join(deleted)}"
    return False, f"策略 v{version} 不存在"


def get_backtest_result(version: int) -> Tuple[Optional[Dict], Optional[str]]:
    """获取指定策略的回测结果"""
    path = PROJECT_ROOT / "results" / f"strategy_v{version}.json"
    if not path.exists():
        return None, f"策略 v{version} 的回测结果不存在"
    with open(path) as f:
        return json.load(f), None


def get_all_results() -> List[Dict]:
    """获取所有回测结果摘要"""
    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))
    output = []
    for r in results:
        with open(r) as f:
            m = json.load(f)
        m["version"] = _extract_version(r.stem)
        output.append(m)
    output.sort(key=lambda x: x.get("version", 0))
    return output


def get_evolution_comparison() -> List[Dict]:
    """获取进化对比数据（所有策略的核心指标）"""
    results_dir = PROJECT_ROOT / "results"
    results = sorted(results_dir.glob("strategy_v*.json"))
    comparison = []
    for r in results:
        with open(r) as f:
            m = json.load(f)
        comparison.append({
            "version": m.get("version", _extract_version(r.stem)),
            "name": m.get("strategy_name", r.stem),
            "total_return": m.get("total_return", 0),
            "annual_return": m.get("annual_return", 0),
            "max_drawdown": m.get("max_drawdown", 0),
            "sharpe_ratio": m.get("sharpe_ratio", 0),
            "calmar_ratio": m.get("calmar_ratio", 0),
            "information_ratio": m.get("information_ratio", 0),
        })
    comparison.sort(key=lambda x: x["version"])
    return comparison


def get_system_status() -> Dict:
    """获取系统配置和状态"""
    try:
        cfg = load_config()
    except Exception:
        cfg = None

    pool_dir = PROJECT_ROOT / "strategy_pool"
    results_dir = PROJECT_ROOT / "results"

    strategies = sorted(pool_dir.glob("strategy_v*.py"))
    results = sorted(results_dir.glob("strategy_v*.json"))

    # 检查 LLM 是否可用
    llm_available = False
    try:
        from llm.deepseek_client import create_client_from_config
        create_client_from_config()
        llm_available = True
    except Exception:
        pass

    return {
        "strategy_count": len(strategies),
        "result_count": len(results),
        "stock_pool": getattr(cfg, "STOCK_POOL", "hs300") if cfg else "hs300",
        "date_range": f"{getattr(cfg, 'DATA_START_DATE', '2020-01-01')} → {getattr(cfg, 'DATA_END_DATE', '2025-12-31')}",
        "initial_capital": getattr(cfg, "INITIAL_CAPITAL", 1_000_000) if cfg else 1_000_000,
        "llm_available": llm_available,
    }


# ============================================================
# 核心操作（异步模拟同步，实际在 FastAPI 中用 BackgroundTasks）
# ============================================================

def run_fetch_data(progress_callback=None) -> Tuple[pd.DataFrame, Optional[str], object]:
    """Step 1: 下载/更新数据"""
    from data.fetcher import DataFetcher

    cfg = load_config()
    fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)

    if progress_callback:
        progress_callback("📊 正在获取成分股并下载日线数据...")

    df = fetcher.fetch_daily(
        stock_pool=cfg.STOCK_POOL,
        start_date=cfg.DATA_START_DATE,
        end_date=cfg.DATA_END_DATE,
    )

    if progress_callback:
        progress_callback(f"✅ 数据就绪: {len(df)} 行, {df['code'].nunique()} 支股票")

    return df, None, fetcher


def run_backtest_for_version(
    version: int,
    progress_callback=None,
) -> Dict:
    """Step 3: 运行单个策略回测"""
    from data.fetcher import DataFetcher
    from engine.sandbox import execute_strategy, load_strategy_from_file
    from engine.backtest import run_backtest_simple
    from engine.analyzer import compute_metrics, save_metrics_json

    cfg = load_config()
    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"

    if not strategy_path.exists():
        return {"error": f"策略文件 strategy_v{version}.py 不存在"}

    if progress_callback:
        progress_callback(f"📦 加载数据...")

    fetcher = DataFetcher(cache_dir=cfg.DATA_CACHE_DIR)

    data = fetcher.get_data(
        start_date=cfg.DATA_START_DATE,
        end_date=cfg.DATA_END_DATE,
        universe=cfg.STOCK_POOL,
    )

    if progress_callback:
        progress_callback(f"🧪 沙箱执行策略 v{version}...")

    signals, error = execute_strategy(strategy_path, data)
    if error:
        return {"error": error, "phase": "sandbox"}

    if progress_callback:
        progress_callback(f"📊 VectorBT 回测中...")

    result = run_backtest_simple(
        data=data,
        signals=signals,
        initial_capital=cfg.INITIAL_CAPITAL,
        commission=cfg.COMMISSION,
        slippage=cfg.SLIPPAGE,
    )

    if "error" in result and result["error"]:
        return {"error": result["error"], "phase": "backtest"}

    # 基准
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

    if progress_callback:
        progress_callback(f"📈 计算绩效指标...")

    equity = result.get("equity_curve", pd.Series(dtype=float))
    trades = result.get("trades")

    metrics = compute_metrics(
        equity_curve=equity,
        trades=trades,
        risk_free_rate=cfg.RISK_FREE_RATE,
        benchmark_returns=benchmark_returns,
    )

    # 附加策略元信息
    strategy, _ = load_strategy_from_file(strategy_path)
    if strategy and hasattr(strategy, "get_info"):
        info = strategy.get_info()
        metrics["strategy_name"] = info.get("name", strategy_path.stem)
        metrics["version"] = int(info.get("version", version))
    else:
        metrics["strategy_name"] = strategy_path.stem
        metrics["version"] = version

    # 保存结果
    result_path = PROJECT_ROOT / "results" / f"strategy_v{version}.json"
    save_metrics_json(metrics, str(result_path))

    if progress_callback:
        progress_callback(f"✅ 回测完成 — 夏普: {metrics.get('sharpe_ratio', 0):.2f}")

    # 生成权益曲线数据（用于前端图表）
    equity_list = []
    if isinstance(equity, pd.Series) and len(equity) > 0:
        if isinstance(equity.index, pd.DatetimeIndex):
            equity_list = [
                {"date": str(d.date()), "value": float(v)}
                for d, v in equity.items()
            ]
        else:
            equity_list = [
                {"index": int(i), "value": float(v)}
                for i, v in enumerate(equity)
            ]

    return {
        "success": True,
        "version": version,
        "metrics": metrics,
        "equity_curve": equity_list[-500:] if len(equity_list) > 500 else equity_list,
    }


def run_generate_strategy(instruction: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Step 2: AI 生成新策略"""
    from llm.deepseek_client import create_client_from_config

    try:
        client = create_client_from_config()
    except ValueError as e:
        return None, str(e)

    code, error = client.generate_strategy(instruction=instruction)
    if error:
        return None, error
    if not code:
        return None, "AI 返回空内容"

    version = _get_next_version()
    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    strategy_path.write_text(code, encoding="utf-8")

    return {"version": version, "code": code, "filename": f"strategy_v{version}.py"}, None


def run_improve_strategy(version: int) -> Tuple[Optional[Dict], Optional[str]]:
    """Step 5: AI 改进策略"""
    from llm.deepseek_client import create_client_from_config
    from engine.analyzer import metrics_to_feedback_context

    result_path = PROJECT_ROOT / "results" / f"strategy_v{version}.json"
    if not result_path.exists():
        return None, f"策略 v{version} 的回测结果不存在，请先运行回测"

    with open(result_path) as f:
        metrics = json.load(f)

    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    if not strategy_path.exists():
        return None, f"策略文件 strategy_v{version}.py 不存在"

    current_code = strategy_path.read_text(encoding="utf-8")
    strategy_name = metrics.get("strategy_name", f"strategy_v{version}")
    feedback = metrics_to_feedback_context(metrics, strategy_name, version)

    try:
        client = create_client_from_config()
    except ValueError as e:
        return None, str(e)

    code, error = client.analyze_and_improve(
        feedback_context=feedback, current_code=current_code
    )
    if error:
        return None, error
    if not code:
        return None, "AI 返回空内容"

    new_version = _get_next_version()
    new_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{new_version}.py"
    new_path.write_text(code, encoding="utf-8")

    return {"version": new_version, "code": code, "filename": f"strategy_v{new_version}.py"}, None


def run_fix_strategy(version: int, error_message: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Step 7: AI 修复策略"""
    from llm.deepseek_client import create_client_from_config

    strategy_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{version}.py"
    if not strategy_path.exists():
        return None, f"策略文件 strategy_v{version}.py 不存在"

    code = strategy_path.read_text(encoding="utf-8")

    try:
        client = create_client_from_config()
    except ValueError as e:
        return None, str(e)

    fixed_code, error = client.fix_code(error_message=error_message, strategy_code=code)
    if error:
        return None, error
    if not fixed_code:
        return None, "AI 返回空内容"

    new_version = _get_next_version()
    new_path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{new_version}.py"
    new_path.write_text(fixed_code, encoding="utf-8")

    return {"version": new_version, "code": fixed_code, "filename": f"strategy_v{new_version}.py"}, None


# ============================================================
# 辅助
# ============================================================

def _extract_version(stem: str) -> int:
    try:
        return int(stem.replace("strategy_v", ""))
    except ValueError:
        return 0


def _get_next_version() -> int:
    pool_dir = PROJECT_ROOT / "strategy_pool"
    existing = list(pool_dir.glob("strategy_v*.py"))
    if not existing:
        return 1
    versions = []
    for f in existing:
        try:
            versions.append(int(f.stem.replace("strategy_v", "")))
        except ValueError:
            continue
    return max(versions) + 1 if versions else 1