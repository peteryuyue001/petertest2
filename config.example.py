"""
量化 AI 进化系统 — 全局配置文件

⚠️ 使用前请复制此文件为 config.py，并填入真实的 API Key：
   cp config.example.py config.py

config.py 已在 .gitignore 中，不会被提交到 GitHub。
"""

# ============================================
# 🔑 DeepSeek API 配置
# ============================================
DEEPSEEK_API_KEY = "sk-your-api-key-here"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # deepseek-chat 或 deepseek-reasoner

# ============================================
# 📊 数据配置
# ============================================
# A股数据起始日期
DATA_START_DATE = "2020-01-01"
DATA_END_DATE = "2025-12-31"

# 股票池：hs300（沪深300）, csi500（中证500）, csi1000（中证1000）
STOCK_POOL = "hs300"

# 本地缓存目录
DATA_CACHE_DIR = "data/cache"

# ============================================
# 💰 回测配置
# ============================================
INITIAL_CAPITAL = 1_000_000.0  # 初始资金（元）
COMMISSION = 0.0003  # 手续费率（万三）
SLIPPAGE = 0.001  # 滑点（千一）

# 无风险利率（用于夏普比率计算）
RISK_FREE_RATE = 0.02  # 2%

# 基准标的（用于信息比率计算）
BENCHMARK_SYMBOL = "000300"  # 沪深300指数

# ============================================
# 🗄 结果存储
# ============================================
RESULTS_DIR = "results"
STRATEGY_POOL_DIR = "strategy_pool"

# ============================================
# 🔄 进化参数
# ============================================
# 初始策略数量
INITIAL_POPULATION_SIZE = 5

# 最大进化代数
MAX_GENERATIONS = 20

# 淘汰阈值：连续此代无改善则停止
EARLY_STOP_AFTER = 5