import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class HS300MultiFactorStrategyV4(BaseStrategy):
    name = "沪深300多因子选股策略 v4"
    version = 4
    description = "基于动量、质量、低波、小市值的周度调仓策略，极端波动择时空仓，等权重持有15支股票"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # ===== 数据预处理 =====
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        if data['date'].dtype != 'object':
            data['date'] = data['date'].astype(str)

        # ===== 因子计算 =====
        # 1. 动量因子（20日涨幅，正方向：强者恒强）
        data['momentum_20'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).sum())

        # 2. 质量因子（ROE近似：用换手率变化率衡量盈利稳定性，这里用20日换手率均值倒数）
        data['turnover_20'] = data.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        data['quality'] = 1 / (data['turnover_20'] + 0.01)  # 低换手率代表高质量

        # 3. 低波因子（20日波动率倒数）
        data['volatility_20'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        data['low_vol'] = 1 / (data['volatility_20'] + 0.001)

        # 4. 小市值因子（使用成交额/成交量得到价格，再乘以流通股数近似市值，这里直接用价格倒数）
        data['price'] = data['amount'] / data['volume'].replace(0, np.nan)
        data['small_cap'] = 1 / (data['price'] + 0.01)  # 价格越低，市值越小

        # ===== 因子处理 =====
        data = data.dropna(subset=['momentum_20', 'quality', 'low_vol', 'small_cap'])

        # 因子标准化（Z-score）
        for factor in ['momentum_20', 'quality', 'low_vol', 'small_cap']:
            mean = data.groupby('date')[factor].transform('mean')
            std = data.groupby('date')[factor].transform('std')
            data[f'{factor}_zscore'] = (data[factor] - mean) / std.replace(0, np.nan)

        # ===== 综合评分（等权合成） =====
        data['score'] = (data['momentum_20_zscore'] 
                         + data['quality_zscore'] 
                         + data['low_vol_zscore'] 
                         + data['small_cap_zscore'])

        # ===== 极端波动择时：当市场20日波动率超过历史90%分位数时空仓 =====
        market_vol = data.groupby('date')['volatility_20'].mean().reset_index()
        market_vol.columns = ['date', 'market_volatility']
        # 计算滚动90%分位数
        market_vol['vol_quantile_90'] = market_vol['market_volatility'].rolling(60).quantile(0.9)
        market_vol = market_vol.dropna(subset=['vol_quantile_90'])
        market_vol['extreme_vol'] = market_vol['market_volatility'] > market_vol['vol_quantile_90']
        data = data.merge(market_vol[['date', 'extreme_vol']], on='date', how='left')

        # ===== 选股逻辑 =====
        # 周度调仓
        data['year_week'] = data['date'].str[:10]  # 取完整日期，后续按周筛选
        data['week_end'] = data.groupby(['code', 'year_week'])['date'].transform('max')
        data['is_week_end'] = data['date'] == data['week_end']

        # 只保留周度末且市场未处于极端波动状态的交易日
        week_end_data = data[data['is_week_end'] & ~data['extreme_vol']].copy()

        # 如果市场处于极端波动状态，持仓为空
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