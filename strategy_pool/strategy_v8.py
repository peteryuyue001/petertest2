import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class HS300MultiFactorStrategyV5(BaseStrategy):
    name = "沪深300多因子选股策略 v5"
    version = 5
    description = "基于反转、低换手、高波、低成交额因子的周度调仓策略，大盘均线趋势过滤，等权重持有15支股票"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # ===== 数据预处理 =====
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        if data['date'].dtype != 'object':
            data['date'] = data['date'].astype(str)

        # ===== 因子计算 =====
        # 1. 反转因子（20日累计跌幅，负方向：跌得多的未来可能涨）
        data['reversal_20'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).sum())
        data['reversal_20'] = -data['reversal_20']  # 取负号，使跌幅大的股票得分高

        # 2. 低换手因子（20日平均换手率倒数，低换手代表关注度低，可能有补涨机会）
        data['turnover_20'] = data.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        data['low_turnover'] = 1 / (data['turnover_20'] + 0.01)

        # 3. 高波动因子（20日波动率，高波动在反弹时弹性更大）
        data['volatility_20'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        data['high_vol'] = data['volatility_20']  # 直接使用波动率

        # 4. 低成交额因子（20日平均成交额倒数，成交额低可能代表被低估）
        data['amount_20'] = data.groupby('code')['amount'].transform(lambda x: x.rolling(20).mean())
        data['low_amount'] = 1 / (data['amount_20'] + 1e6)  # 加一个较大的数防止除零

        # ===== 因子处理 =====
        data = data.dropna(subset=['reversal_20', 'low_turnover', 'high_vol', 'low_amount'])

        # 因子标准化（Z-score）
        for factor in ['reversal_20', 'low_turnover', 'high_vol', 'low_amount']:
            mean = data.groupby('date')[factor].transform('mean')
            std = data.groupby('date')[factor].transform('std')
            data[f'{factor}_zscore'] = (data[factor] - mean) / std.replace(0, np.nan)

        # ===== 综合评分（等权合成） =====
        data['score'] = (data['reversal_20_zscore'] 
                         + data['low_turnover_zscore'] 
                         + data['high_vol_zscore'] 
                         + data['low_amount_zscore'])

        # ===== 大盘趋势过滤：沪深300指数价格在20日均线之上才持仓 =====
        # 取所有股票中成交额最大的作为大盘近似（或者直接使用指数数据，这里简化处理）
        # 这里我们使用所有股票的平均价格作为大盘趋势的近似，或者更简单：只保留有沪深300ETF数据的日期
        # 由于data中没有指数数据，我们用所有股票的平均价格来模拟
        daily_market_data = data.groupby('date').agg(
            avg_price=('close', 'mean')
        ).reset_index()
        daily_market_data['ma20'] = daily_market_data['avg_price'].rolling(20).mean()
        daily_market_data['trend_up'] = daily_market_data['avg_price'] > daily_market_data['ma20']
        data = data.merge(daily_market_data[['date', 'trend_up']], on='date', how='left')

        # ===== 选股逻辑 =====
        # 周度调仓
        data['year_week'] = data['date'].str[:10]  # 取完整日期，后续按周筛选
        data['week_end'] = data.groupby(['code', 'year_week'])['date'].transform('max')
        data['is_week_end'] = data['date'] == data['week_end']

        # 只保留周度末且大盘趋势向上的交易日
        week_end_data = data[data['is_week_end'] & data['trend_up']].copy()

        # 如果大盘趋势向下，持仓为空
        if week_end_data.empty:
            return pd.DataFrame(columns=['code', 'weight', 'date'])

        # 在每周末，按评分排序选前15支股票
        def select_stocks(group):
            group = group.sort_values('score', ascending=False)
            selected = group.head(15)
            selected['weight'] = 1.0 / len(selected)
            return selected

        selected = week_end_data.groupby('date', group_keys=False).apply(select_stocks)

        # ===== 生成信号 =====
        signals_df = selected[['code', 'weight', 'date']].reset_index(drop=True)
        signals_df['weight'] = signals_df['weight'].clip(0, 0.2)

        if signals_df.empty:
            return pd.DataFrame(columns=['code', 'weight', 'date'])

        return signals_df