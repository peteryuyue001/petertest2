import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class HS300MultiFactorStrategyV3(BaseStrategy):
    name = "沪深300多因子选股策略 v3"
    version = 3
    description = "基于反转、波动率、估值、换手率的周度调仓策略，波动率择时过滤，等权重持有15支股票"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # ===== 数据预处理 =====
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        if data['date'].dtype != 'object':
            data['date'] = data['date'].astype(str)

        # ===== 因子计算 =====
        # 1. 5日反转因子（负值：短期超跌反弹）
        data['reversal_5'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(5).sum())

        # 2. 波动率因子（负值：低波动股票更稳定）
        data['volatility_20'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())

        # 3. 换手率因子（负值：低换手率代表筹码稳定）
        data['turnover_20'] = data.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())

        # 4. 估值因子（使用市净率近似：价格/每股净资产，这里用价格/（成交额/成交量）作为代理）
        data['pb_ratio'] = data['close'] / (data['amount'] / data['volume']).replace(0, np.nan) * 100
        data['pb_ratio'] = data['pb_ratio'].clip(lower=0.1, upper=100)

        # ===== 因子处理 =====
        data = data.dropna(subset=['reversal_5', 'volatility_20', 'turnover_20', 'pb_ratio'])

        # 因子标准化（Z-score）
        for factor in ['reversal_5', 'volatility_20', 'turnover_20', 'pb_ratio']:
            mean = data.groupby('date')[factor].transform('mean')
            std = data.groupby('date')[factor].transform('std')
            data[f'{factor}_zscore'] = (data[factor] - mean) / std.replace(0, np.nan)

        # ===== 综合评分 =====
        # 反转因子取负值（超跌反弹）
        # 波动率因子取负值（低波动稳定）
        # 换手率因子取负值（低换手筹码稳定）
        # 估值因子取负值（低估值安全边际高）
        data['score'] = (-data['reversal_5_zscore'] 
                         - data['volatility_20_zscore'] 
                         - data['turnover_20_zscore'] 
                         - data['pb_ratio_zscore'])

        # ===== 波动率择时：20日波动率过滤 =====
        # 计算市场整体波动率（使用所有股票的平均波动率）
        market_vol = data.groupby('date')['volatility_20'].mean().reset_index()
        market_vol.columns = ['date', 'market_volatility']
        market_vol['vol_ma20'] = market_vol['market_volatility'].rolling(20).mean()
        market_vol = market_vol.dropna(subset=['vol_ma20'])
        # 当市场波动率低于20日均值时，认为市场稳定，可以持仓
        market_vol['low_volatility'] = market_vol['market_volatility'] < market_vol['vol_ma20']
        data = data.merge(market_vol[['date', 'low_volatility']], on='date', how='left')

        # ===== 选股逻辑 =====
        # 使用字符串切片提取年月日，实现周度调仓
        data['year_week'] = data['date'].str[:10]  # 取完整日期，后续按周筛选

        # 标记每周最后一个交易日
        data['week_end'] = data.groupby(['code', 'year_week'])['date'].transform('max')
        data['is_week_end'] = data['date'] == data['week_end']

        # 只保留周度末且市场处于低波动状态的交易日
        week_end_data = data[data['is_week_end'] & data['low_volatility']].copy()

        # 如果市场处于高波动状态，持仓为空
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

        # 如果波动率择时导致无信号，返回空DataFrame（保持与框架兼容）
        if signals_df.empty:
            return pd.DataFrame(columns=['code', 'weight', 'date'])

        return signals_df