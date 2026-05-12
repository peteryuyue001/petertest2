"""
数据获取模块 — 基于 AkShare 的 A 股日线数据接口

功能：
  - 获取沪深300/中证500/中证1000 成分股
  - 下载日线行情（复权价）
  - 本地 Parquet 缓存，避免重复下载
  - 输出标准化 DataFrame 供策略使用
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# 将项目根目录添加到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False


# 默认缓存目录
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"

# A股交易日通常每年约 244 天
TRADING_DAYS_PER_YEAR = 244


class DataFetcher:
    """
    A 股数据获取器

    Usage:
        fetcher = DataFetcher()
        df = fetcher.fetch_daily(stock_pool="hs300",
                                 start_date="2020-01-01",
                                 end_date="2025-12-31")
    """

    # 指数成分股映射
    INDEX_MAP = {
        "hs300": ("000300", "沪深300"),
        "csi500": ("000905", "中证500"),
        "csi1000": ("000852", "中证1000"),
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: 缓存目录路径，默认 data/cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_AKSHARE:
            raise ImportError(
                "akshare 未安装，请运行: pip install akshare"
            )

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def fetch_daily(
        self,
        stock_pool: str = "hs300",
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        获取指定股票池的日线数据（后复权）

        Args:
            stock_pool: 股票池名称 (hs300 / csi500 / csi1000)
            start_date: 起始日期 YYYY-MM-DD
            end_date: 截止日期 YYYY-MM-DD

        Returns:
            DataFrame with columns:
                date, code, name, open, high, low, close, volume, amount,
                turnover_rate, pct_change
        """
        if stock_pool not in self.INDEX_MAP:
            raise ValueError(
                f"不支持的股票池: {stock_pool}，"
                f"可选: {list(self.INDEX_MAP.keys())}"
            )

        # 尝试从缓存加载
        cache_path = self._cache_path(stock_pool, start_date, end_date)
        if cache_path.exists():
            print(f"📦 从缓存加载数据: {cache_path}")
            df = pd.read_parquet(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            return df

        # 下载数据
        index_code, index_name = self.INDEX_MAP[stock_pool]
        print(f"📊 获取 {index_name}({index_code}) 成分股...")

        stocks = self._get_constituents(index_code, index_name)
        if stocks.empty:
            raise RuntimeError(f"无法获取 {index_name} 成分股列表")

        print(f"   成分股数量: {len(stocks)}")

        # 批量下载日线
        all_data = []
        failed_codes = []

        for idx, row in stocks.iterrows():
            code = row["code"]
            name = row.get("name", "")

            try:
                df_stock = self._download_one_stock(
                    code, name, start_date, end_date
                )
                if not df_stock.empty:
                    all_data.append(df_stock)
            except Exception as e:
                failed_codes.append(code)
                if len(failed_codes) <= 5:
                    print(f"   ⚠️ {code} {name} 下载失败: {e}")

            # 进度提示（每 50 支打印一次）
            if (idx + 1) % 50 == 0:
                print(f"   已下载 {idx + 1}/{len(stocks)} ...")

        if failed_codes:
            print(f"   ⚠️ 共 {len(failed_codes)} 支股票下载失败")

        if not all_data:
            raise RuntimeError("没有成功下载任何股票数据")

        df = pd.concat(all_data, ignore_index=True)

        # 日期过滤
        df["date"] = pd.to_datetime(df["date"])
        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        # 排序
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        # 写入缓存
        df.to_parquet(cache_path, index=False)
        print(f"✅ 数据已缓存: {cache_path} ({len(df)} 行)")

        return df

    def fetch_benchmark(
        self,
        symbol: str = "000300",
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        获取基准指数日线（用于信息比率计算）

        Args:
            symbol: 指数代码 (000300 沪深300, 000905 中证500)
            start_date: 起始日期
            end_date: 截止日期

        Returns:
            DataFrame with columns: date, close, pct_change
        """
        cache_path = self.cache_dir / f"benchmark_{symbol}_{start_date}_{end_date}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            return df

        print(f"📈 获取基准指数 {symbol} 日线...")

        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
        except Exception:
            try:
                df = ak.stock_zh_index_daily(symbol=f"sz{symbol}")
            except Exception as e:
                raise RuntimeError(f"获取基准指数失败: {e}")

        df = df.rename(columns={"date": "date", "close": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # 计算日收益率
        df["pct_change"] = df["close"].pct_change()

        df = df[
            (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        df.to_parquet(cache_path, index=False)
        print(f"✅ 基准数据已缓存: {cache_path}")

        return df

    def get_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
        universe: str = "hs300",
    ) -> pd.DataFrame:
        """
        标准数据接口（供 AI 生成的策略调用）

        这是策略代码中 get_data() 函数的唯一数据源。
        返回已处理好的 DataFrame，可直接用于因子计算。
        """
        return self.fetch_daily(
            stock_pool=universe,
            start_date=start_date,
            end_date=end_date,
        )

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _get_constituents(
        self, index_code: str, index_name: str
    ) -> pd.DataFrame:
        """获取指数成分股列表"""
        try:
            if "300" in index_code:
                df = ak.index_stock_cons_csindex(symbol=index_code)
            elif "905" in index_code:
                df = ak.index_stock_cons_csindex(symbol=index_code)
            elif "852" in index_code:
                df = ak.index_stock_cons_csindex(symbol=index_code)
            else:
                df = ak.index_stock_cons(symbol=index_code)
        except Exception:
            # 回退方案：使用 stock_zh_a_spot_em 按市值筛选
            try:
                df_spot = ak.stock_zh_a_spot_em()
                df_spot = df_spot.sort_values("总市值", ascending=False)
                n = 300 if "300" in index_code else 500
                df = df_spot.head(n)[["代码", "名称"]].copy()
                df.columns = ["code", "name"]
                return df
            except Exception as e:
                raise RuntimeError(f"获取成分股失败: {e}")

        # 统一列名
        if "品种代码" in df.columns:
            df = df.rename(columns={"品种代码": "code", "品种名称": "name"})
        elif "成分券代码" in df.columns:
            df = df.rename(
                columns={"成分券代码": "code", "成分券名称": "name"}
            )
        elif "constituent_code" in df.columns:
            df = df.rename(
                columns={"constituent_code": "code", "constituent_name": "name"}
            )

        if "code" not in df.columns:
            # 尝试自动查找包含 "code" 或 "代码" 的列
            code_cols = [c for c in df.columns if "代码" in c or "code" in c.lower()]
            name_cols = [c for c in df.columns if "名称" in c or "name" in c.lower()]
            if code_cols and name_cols:
                df = df[[code_cols[0], name_cols[0]]].copy()
                df.columns = ["code", "name"]

        return df[["code", "name"]].drop_duplicates().reset_index(drop=True)

    def _download_one_stock(
        self, code: str, name: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """下载单支股票日线数据"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="hfq",  # 后复权
            )
        except Exception:
            # 部分新上市股票可能 ak 接口不支持，静默跳过
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # 标准化列名
        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover_rate",
                "涨跌幅": "pct_change",
            }
        )

        df["code"] = code
        df["name"] = name

        keep_cols = [
            "date", "code", "name", "open", "high", "low",
            "close", "volume", "amount", "turnover_rate", "pct_change",
        ]
        df = df[[c for c in keep_cols if c in df.columns]]

        return df

    def _cache_path(
        self, stock_pool: str, start_date: str, end_date: str
    ) -> Path:
        """生成缓存文件路径"""
        filename = f"{stock_pool}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename


# ============================================================
# 模块级快捷函数（供 AI 生成的策略直接 import）
# ============================================================

_fetcher: Optional[DataFetcher] = None


def get_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    universe: str = "hs300",
) -> pd.DataFrame:
    """
    标准数据接口 — AI 生成的策略必须使用此函数获取数据

    Args:
        start_date: 回测起始日期
        end_date: 回测截止日期
        universe: 股票池 (hs300 / csi500 / csi1000)

    Returns:
        DataFrame with columns:
            date, code, name, open, high, low, close, volume, amount,
            turnover_rate, pct_change
    """
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher.get_data(start_date, end_date, universe)