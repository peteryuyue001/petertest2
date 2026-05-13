import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class HS300MultiFactorStrategyV2(BaseStrategy):
    name = "沪深300多因子选股策略 v2"
    version = 2
    description = "基于20日动量、波动率、换手率、市值的周度调仓策略，大盘择时过滤，等权重持有10支股票"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # ===== 数据预处理 =====
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        if data['date'].dtype != 'object':
            data['date'] = data['date'].astype(str)

        # ===== 因子计算 =====
        # 1. 20日动量因子（正值：追涨强势股）
        data['momentum_20'] = data.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)

        # 2. 波动率因子（正值：高波动股票弹性更大）
        data['volatility_20'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())

        # 3. 换手率因子（负值：低换手率代表筹码稳定）
        data['turnover_20'] = data.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())

        # 4. 小市值因子（正值：沪深300内市值小的股票弹性更大）
        # 使用日频数据中的估值指标近似，这里用成交量*价格作为市值代理
        data['market_value'] = data['close'] * data['volume'] / 100000000  # 单位：亿元
        # 取对数平滑
        data['log_market_value'] = np.log(data['market_value'].clip(lower=1))

        # ===== 因子处理 =====
        data = data.dropna(subset=['momentum_20', 'volatility_20', 'turnover_20', 'log_market_value'])

        # 因子标准化（Z-score）
        for factor in ['momentum_20', 'volatility_20', 'turnover_20', 'log_market_value']:
            mean = data.groupby('date')[factor].transform('mean')
            std = data.groupby('date')[factor].transform('std')
            data[f'{factor}_zscore'] = (data[factor] - mean) / std.replace(0, np.nan)

        # ===== 综合评分 =====
        # 动量因子取正值（追涨强势）
        # 波动率因子取正值（高波动弹性）
        # 换手率因子取负值（低换手筹码稳定）
        # 市值因子取负值（小市值弹性好）
        data['score'] = (data['momentum_20_zscore'] 
                         + data['volatility_20_zscore'] 
                         - data['turnover_20_zscore'] 
                         - data['log_market_value_zscore'])

        # ===== 大盘择时：20日均线过滤 =====
        # 计算沪深300指数（使用所有股票的平均价格作为代理，更精确需使用指数数据）
        index_data = data.groupby('date')['close'].mean().reset_index()
        index_data.columns = ['date', 'index_close']
        index_data['index_ma20'] = index_data['index_close'].rolling(20).mean()
        index_data = index_data.dropna(subset=['index_ma20'])
        index_data['index_above_ma20'] = index_data['index_close'] > index_data['index_ma20']
        data = data.merge(index_data[['date', 'index_above_ma20']], on='date', how='left')

        # ===== 选股逻辑 =====
        # 使用字符串切片提取年月日，实现周度调仓
        data['year_week'] = data['date'].str[:10]  # 取完整日期，后续按周筛选

        # 标记每周最后一个交易日
        data['week_end'] = data.groupby(['code', 'year_week'])['date'].transform('max')
        data['is_week_end'] = data['date'] == data['week_end']

        # 只保留周度末且大盘处于20日均线上方的交易日
        week_end_data = data[data['is_week_end'] & data['index_above_ma20']].copy()

        # 如果大盘处于均线下方，持仓为空
        if week_end_data.empty:
            return pd.DataFrame(columns=['code', 'weight', 'date'])

        # 在每周末，按评分排序选前10支股票
        def select_stocks(group):
            group = group.sort_values('score', ascending=False)
            selected = group.head(10)
            selected['weight'] = 1.0 / len(selected)
            return selected

        selected = week_end_data.groupby('date', group_keys=False).apply(select_stocks)

        # ===== 生成信号 =====
        signals_df = selected[['code', 'weight', 'date']].reset_index(drop=True)
        signals_df['weight'] = signals_df['weight'].clip(0, 0.2)

        # 如果大盘择时导致无信号，返回空DataFrame（保持与框架兼容）
        if signals_df.empty:
            return pd.DataFrame(columns=['code', 'weight', 'date'])

        return signals_df