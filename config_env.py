"""
Railway/云端部署用配置 — 从环境变量读取敏感信息
当 config.py 不存在时，bridge.py 会 fallback 到 config.example.py，
但在 Railway 上我们通过此文件注入环境变量。
"""
import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

DATA_START_DATE = os.environ.get("DATA_START_DATE", "2020-01-01")
DATA_END_DATE = os.environ.get("DATA_END_DATE", "2025-12-31")
STOCK_POOL = os.environ.get("STOCK_POOL", "hs300")
DATA_CACHE_DIR = os.environ.get("DATA_CACHE_DIR", "data/cache")

INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000000"))
COMMISSION = float(os.environ.get("COMMISSION", "0.0003"))
SLIPPAGE = float(os.environ.get("SLIPPAGE", "0.001"))
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.02"))
BENCHMARK_SYMBOL = os.environ.get("BENCHMARK_SYMBOL", "000300.SH")
