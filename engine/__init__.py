"""Engine package for backtesting and analysis."""
from .backtest import run_backtest, run_backtest_simple, prepare_price_matrix, prepare_signal_matrix
from .analyzer import compute_metrics, format_metrics_report, metrics_to_feedback_context, save_metrics_json
from .sandbox import execute_strategy, load_strategy_from_file, format_error_for_llm

__all__ = [
    "run_backtest",
    "run_backtest_simple",
    "prepare_price_matrix",
    "prepare_signal_matrix",
    "compute_metrics",
    "format_metrics_report",
    "metrics_to_feedback_context",
    "save_metrics_json",
    "execute_strategy",
    "load_strategy_from_file",
    "format_error_for_llm",
]
