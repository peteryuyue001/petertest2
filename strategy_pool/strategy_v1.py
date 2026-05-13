import pandas as pd
from template.strategy_base import BaseStrategy


class ExampleMomentumStrategy(BaseStrategy):
    """
    简单动量策略示例

    每月最后一个交易日按20日收益率选出20支股票，等权配置。
    """

    name = "示例动量策略"
    version = 1
    description = "基于20日动量的简单选股策略"

    def generate_signals(self, data):
        data = data.sort_values(["code", "date"]).copy()

        data["return_20d"] = data.groupby("code")["close"].transform(
            lambda x: x.pct_change(periods=20)
        )

        data["month"] = data["date"].dt.to_period("M")
        rebalance_dates = data.groupby("month")["date"].transform("max")
        mask = data["date"] == rebalance_dates

        rebalance_data = data[mask].copy()

        signals_list = []
        for dt in sorted(rebalance_data["date"].unique()):
            day_data = rebalance_data[rebalance_data["date"] == dt].copy()
            day_data = day_data.dropna(subset=["return_20d"])
            day_data = day_data.nlargest(20, "return_20d")
            if len(day_data) == 0:
                continue
            day_data["weight"] = 1.0 / len(day_data)
            signals_list.append(day_data[["code", "weight", "date"]])

        if not signals_list:
            return pd.DataFrame(columns=["code", "weight", "date"])

        signals = pd.concat(signals_list, ignore_index=True)
        return self.validate_output(signals)
