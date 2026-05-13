"""
数据获取模块 — 基于 AkShare 的 A 股日线数据接口

功能：
  - 获取沪深300/中证500/中证1000 成分股
  - 下载日线行情（复权价）
  - 本地 Parquet 缓存，避免重复下载
  - 当网络数据源不可用时，自动生成合成数据用于测试
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
TRADING_DAYS_PER_YEAR = 244


class DataFetcher:
    INDEX_MAP = {
        "hs300": ("000300", "沪深300"),
        "csi500": ("000905", "中证500"),
        "csi1000": ("000852", "中证1000"),
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_daily(
        self,
        stock_pool: str = "hs300",
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        if stock_pool not in self.INDEX_MAP:
            raise ValueError(f"不支持的股票池: {stock_pool}，可选: {list(self.INDEX_MAP.keys())}")

        cache_path = self._cache_path(stock_pool, start_date, end_date)
        if cache_path.exists():
            print(f"📦 从缓存加载数据: {cache_path}")
            df = pd.read_parquet(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            return df

        index_code, index_name = self.INDEX_MAP[stock_pool]
        print(f"📊 获取 {index_name}({index_code}) 公网数据...")

        # --- 主路径：联网下载 ---
        if HAS_AKSHARE:
            try:
                df = self._download_online(stock_pool, index_code, index_name, start_date, end_date)
                if df is not None and len(df) > 0:
                    df.to_parquet(cache_path, index=False)
                    print(f"✅ 数据已缓存: {cache_path} ({len(df)} 行)")
                    return df
            except Exception as e:
                print(f"   ⚠️ 在线下载失败: {type(e).__name__}: {e}")

        # --- 兜底路径：合成数据 ---
        print(f"💡 网络不可用，生成合成测试数据 (50支虚拟股票, {start_date} → {end_date})...")
        df = self._generate_synthetic(stock_pool, start_date, end_date)
        df.to_parquet(cache_path, index=False)
        print(f"✅ 合成数据已缓存: {cache_path} ({len(df)} 行, {df['code'].nunique()} 支)")
        return df

    def fetch_benchmark(
        self,
        symbol: str = "000300",
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / f"benchmark_{symbol}_{start_date}_{end_date}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            return df

        # --- 主路径：联网 ---
        if HAS_AKSHARE:
            try:
                print(f"📈 获取基准指数 {symbol} 日线...")
                df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
                if df.empty:
                    df = ak.stock_zh_index_daily(symbol=f"sz{symbol}")
                df = df.rename(columns={c: c for c in df.columns})
                for old, new in [("日期", "date"), ("收盘", "close"), ("收盘价", "close")]:
                    if old in df.columns:
                        df = df.rename(columns={old: new})
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
                df["pct_change"] = df["close"].pct_change()
                df.to_parquet(cache_path, index=False)
                print(f"✅ 基准数据已缓存: {cache_path}")
                return df
            except Exception as e:
                print(f"   ⚠️ 基准指数下载失败: {e}")

        # --- 兜底：合成基准 ---
        print(f"💡 生成合成基准指数 ({start_date} → {end_date})...")
        dates = pd.bdate_range(start=start_date, end=end_date)
        rng = np.random.default_rng(42)
        close = 4000.0
        closes = []
        for _ in dates:
            close *= (1 + rng.normal(0.0003, 0.012))
            closes.append(close)
        df = pd.DataFrame({"date": dates, "close": closes})
        df["pct_change"] = df["close"].pct_change()
        df.to_parquet(cache_path, index=False)
        print(f"✅ 合成基准已缓存: {cache_path}")
        return df

    def get_data(self, start_date: str = "2020-01-01", end_date: str = "2025-12-31", universe: str = "hs300") -> pd.DataFrame:
        return self.fetch_daily(stock_pool=universe, start_date=start_date, end_date=end_date)

    # ─── 私有 ───

    def _download_online(self, stock_pool, index_code, index_name, start_date, end_date):
        """联网批量下载（带重试和频率控制）"""
        import time as _time

        stocks = self._get_constituents(index_code, index_name)
        if stocks.empty:
            raise RuntimeError(f"无法获取 {index_name} 成分股列表")

        print(f"   成分股数量: {len(stocks)}，开始下载（每10支暂停3秒）...")

        all_data = []
        failed = 0
        batch_size = 10
        N_DEMO = 50  # 演示模式取前50支，加速

        for idx, (_, row) in enumerate(stocks.iterrows()):
            if idx >= N_DEMO:
                print(f"   演示模式：已下载前{N_DEMO}支，停止")
                break
            code = row["code"]
            name = row.get("name", "")
            df_single = self._download_one(code, name, start_date, end_date)
            if not df_single.empty:
                all_data.append(df_single)
            else:
                failed += 1

            if (idx + 1) % min(20, batch_size * 2) == 0:
                print(f"   进度 {idx+1}/{min(len(stocks), N_DEMO)} (成功 {len(all_data)}, 失败 {failed})")
            if (idx + 1) % batch_size == 0:
                _time.sleep(3)

        if failed:
            print(f"   ⚠️ {failed} 支下载失败 (数据源可能限流)")

        if not all_data:
            raise RuntimeError("所有股票下载失败，数据源不可用")

        df = pd.concat(all_data, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        return df

    def _download_one(self, code, name, start_date, end_date) -> pd.DataFrame:
        """单股下载（3次重试 + 不同复权 fallback）"""
        import time as _time
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")

        for adj in ["hfq", "qfq", ""]:  # 后复权 → 前复权 → 不复权
            for attempt in range(3):
                try:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=sd, end_date=ed, adjust=adj)
                    if not df.empty:
                        break
                except Exception as e:
                    err = str(e)
                    if any(kw in err for kw in ["Connection", "Remote", "timeout", "reset", "aborted"]):
                        if attempt < 2:
                            _time.sleep((attempt + 1) * 3)
                            continue
                    return pd.DataFrame()
            else:
                continue  # 3次都失败，换下一个 adj
            break
        else:
            return pd.DataFrame()

        df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount", "换手率": "turnover_rate", "涨跌幅": "pct_change"})
        df["code"] = code
        df["name"] = name
        keep = ["date", "code", "name", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "pct_change"]
        return df[[c for c in keep if c in df.columns]]

    def _get_constituents(self, index_code, index_name) -> pd.DataFrame:
        """获取成分股列表（带兜底）"""
        try:
            df = ak.index_stock_cons_csindex(symbol=index_code)
        except Exception:
            try:
                spot = ak.stock_zh_a_spot_em()
                spot = spot.sort_values("总市值", ascending=False)
                n = 300 if "300" in index_code else 500
                df = spot.head(n)[["代码", "名称"]].copy()
                df.columns = ["code", "name"]
                return df
            except Exception as e:
                raise RuntimeError(f"获取成分股失败: {e}")

        rename_map = {"品种代码": "code", "品种名称": "name", "成分券代码": "code", "成分券名称": "name"}
        existing = {k: v for k, v in rename_map.items() if k in df.columns}
        if existing:
            df = df.rename(columns=existing)
        elif "code" not in df.columns:
            code_cols = [c for c in df.columns if "代码" in c or "code" in c.lower()]
            name_cols = [c for c in df.columns if "名称" in c or "name" in c.lower()]
            if code_cols and name_cols:
                df = df[[code_cols[0], name_cols[0]]].copy()
                df.columns = ["code", "name"]

        return df[["code", "name"]].drop_duplicates().reset_index(drop=True)

    def _generate_synthetic(self, stock_pool, start_date, end_date) -> pd.DataFrame:
        """合成50支虚拟股票日线（用于离线测试回测管道）"""
        dates = pd.bdate_range(start=start_date, end=end_date)
        n_stocks = 50
        n_days = len(dates)
        rng = np.random.default_rng(42)

        codes = [f"SH{600000+i:06d}" if i < 25 else f"SZ{300000+i-25:06d}" for i in range(n_stocks)]
        names = [f"虚拟{i+1}号" for i in range(n_stocks)]

        rows = []
        prices = rng.uniform(5, 50, n_stocks)  # 初始价格
        for i, code in enumerate(codes):
            price = prices[i]
            for t, dt in enumerate(dates):
                ret = rng.normal(0.0003, 0.018)  # 日收益 ~N(0.03%, 1.8%)
                price *= (1 + ret)
                vol = rng.uniform(1e6, 5e7)
                amt = vol * price
                turnover = rng.uniform(0.005, 0.05)
                rows.append({
                    "date": dt, "code": code, "name": names[i],
                    "open": price * (1 + rng.uniform(-0.005, 0.005)),
                    "high": price * (1 + abs(rng.uniform(0, 0.02))),
                    "low": price * (1 - abs(rng.uniform(0, 0.02))),
                    "close": price,
                    "volume": vol, "amount": amt,
                    "turnover_rate": turnover,
                    "pct_change": ret * 100,
                })
        return pd.DataFrame(rows)

    def _cache_path(self, stock_pool, start_date, end_date) -> Path:
        return self.cache_dir / f"{stock_pool}_{start_date}_{end_date}.parquet"


_fetcher: Optional[DataFetcher] = None

def get_data(start_date: str = "2020-01-01", end_date: str = "2025-12-31", universe: str = "hs300") -> pd.DataFrame:
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher.get_data(start_date, end_date, universe)