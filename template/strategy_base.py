"""
策略标准模板 — 约束 AI 生成代码的输入/输出格式

AI 生成的策略必须继承此基类并实现 generate_signals() 方法。

=== 标准接口规范 ===

输入：
    data: pd.DataFrame — 通过 data.fetcher.get_data() 获取
        包含列: date, code, name, open, high, low, close, volume,
               amount, turnover_rate, pct_change

输出：
    pd.DataFrame with columns ['code', 'weight', 'date']
        - code:   股票代码
        - weight: 持仓权重 (0.0 ~ 1.0, 总和 ≤ 1.0)
        - date:   调仓日期

=== 约束规则 ===
1. AI 只能使用 data 参数和 Python 标准库 + numpy/pandas/scipy
2. 不允许访问网络、文件系统
3. 不允许使用 exec/eval
4. 必须返回标准 DataFrame
5. 持仓股票 5~30 支，单支持仓上限 20%
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BaseStrategy(ABC):
    """
    策略基类 — 所有 AI 生成的策略必须继承此类

    Usage:
        class MyStrategy(BaseStrategy):
            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                # AI 编写的因子逻辑
                ...
                return signals_df
    """

    # 策略元信息
    name: str = "BaseStrategy"
    version: int = 1
    description: str = "策略基类"

    # 风控参数
    MAX_STOCKS = 30          # 最大持仓数
    MIN_STOCKS = 5           # 最小持仓数
    MAX_SINGLE_WEIGHT = 0.2  # 单支持仓上限

    @abstractmethod
    def generate_signals(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        生成持仓信号

        Args:
            data: 日线行情数据，必须包含:
                  date, code, open, high, low, close, volume,
                  turnover_rate, pct_change

        Returns:
            DataFrame with columns:
                - code:   股票代码
                - weight: 目标权重 (0.0 ~ 1.0)
                - date:   调仓日期
        """
        ...

    def validate_output(
        self, signals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        验证并清洗 AI 输出，确保符合规范

        Args:
            signals: AI 生成的原始信号 DataFrame

        Returns:
            清洗后的 DataFrame

        Raises:
            ValueError: 输出不符合规范时抛出
        """
        # 1. 检查必要列
        required_cols = {"code", "weight", "date"}
        missing = required_cols - set(signals.columns)
        if missing:
            raise ValueError(
                f"策略 {self.name} 输出缺少必要列: {missing}"
            )

        # 2. 类型转换
        signals = signals.copy(deep=True)
        signals["code"] = signals["code"].astype(str).str.strip()
        signals["weight"] = pd.to_numeric(signals["weight"], errors="coerce")
        signals["date"] = pd.to_datetime(signals["date"], errors="coerce")

        # 3. 清理无效数据
        signals = signals.dropna(subset=["code", "weight", "date"])
        signals = signals[signals["weight"] > 0]

        if signals.empty:
            return pd.DataFrame(columns=["code", "weight", "date"])

        # 4. 权重归一化（保证总和 ≤ 1.0）
        normalized = []
        for dt, group in signals.groupby("date"):
            group = group.copy()
            total = group["weight"].sum()
            if total > 1.0:
                group["weight"] = group["weight"] / total
            normalized.append(group)

        signals = pd.concat(normalized, ignore_index=True)

        # 5. 应用单支持仓上限
        signals["weight"] = signals["weight"].clip(upper=self.MAX_SINGLE_WEIGHT)

        # 6. 限制持仓数量（取权重最高的前 N 支）
        limited = []
        for dt, group in signals.groupby("date"):
            group = group.nlargest(self.MAX_STOCKS, "weight").copy()
            total = group["weight"].sum()
            if total > 1.0:
                group["weight"] = group["weight"] / total
            limited.append(group)

        signals = pd.concat(limited, ignore_index=True)

        return signals[["code", "weight", "date"]]

    def get_info(self) -> Dict[str, str]:
        """返回策略元信息"""
        return {
            "name": self.name,
            "version": str(self.version),
            "description": self.description,
        }


# ============================================================
# 示例策略（演示标准写法）
# ============================================================

class ExampleMomentumStrategy(BaseStrategy):
    """
    示例：动量策略

    选股逻辑：
        - 计算每支股票的 20 日收益率
        - 选择收益率最高的 20 支
        - 等权重配置
    """

    name = "动量策略示例"
    version = 1
    description = "基于20日收益率的简单动量选股策略"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 计算 20 日收益率
        data = data.sort_values(["code", "date"])

        # 每支股票的收益率
        data["return_20d"] = data.groupby("code")["close"].transform(
            lambda x: x.pct_change(periods=20)
        )

        # 每月最后一个交易日调仓
        data["month"] = data["date"].dt.to_period("M")
        # 获取每月最后一天
        rebalance_dates = data.groupby("month")["date"].transform("max")
        mask = data["date"] == rebalance_dates

        # 选股：动量最高的 20 支
        rebalance_data = data[mask].copy()

        signals_list = []
        for dt in rebalance_data["date"].unique():
            day_data = rebalance_data[rebalance_data["date"] == dt].copy()
            day_data = day_data.dropna(subset=["return_20d"])
            day_data = day_data.nlargest(20, "return_20d")

            if len(day_data) == 0:
                continue

            # 等权重
            day_data["weight"] = 1.0 / len(day_data)
            day_data = day_data[["code", "weight", "date"]]
            signals_list.append(day_data)

        if not signals_list:
            return pd.DataFrame(columns=["code", "weight", "date"])

        signals = pd.concat(signals_list, ignore_index=True)
        return self.validate_output(signals)