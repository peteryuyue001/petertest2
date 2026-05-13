import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class PurifiedMultiFactorStrategy(BaseStrategy):
    name = "净化多因子选股策略"
    version = 1
    description = "基于多因子等权合成的月度选股策略，结合反转、小市值、低波动和低换手因子"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 确保数据按股票代码和日期排序
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 计算各因子
        # 1. 反转因子：过去20日收益率（负值表示反转，值越小越好）
        data['momentum_20'] = data.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        data['reversal_factor'] = -data['momentum_20']  # 反转因子：负动量
        
        # 2. 小市值因子：流通市值（用价格*流通股本近似，这里用成交量*均价替代，实际需用市值数据）
        # 由于没有直接市值数据，使用成交额作为市值替代（大市值通常成交额大）
        data['market_cap_proxy'] = data['amount'] / (data['turnover_rate'] + 0.001)  # 估算流通市值
        # 市值排名转化为因子（小市值得分高）
        data['size_factor'] = -data.groupby('date')['market_cap_proxy'].rank(pct=True)
        
        # 3. 低波动因子：过去20日收益率标准差（负值，波动越小越好）
        data['volatility_20'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        data['low_vol_factor'] = -data['volatility_20']
        
        # 4. 低换手因子：过去20日平均换手率（负值，换手越低越好）
        data['turnover_20'] = data.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        data['low_turnover_factor'] = -data['turnover_20']
        
        # 5. 均线偏离因子（控制回撤）：价格相对于60日均线的偏离度，负值表示超跌
        data['ma60'] = data.groupby('code')['close'].transform(lambda x: x.rolling(60).mean())
        data['ma_deviation'] = data['close'] / data['ma60'] - 1
        data['ma_filter'] = np.where(data['ma_deviation'] < 0, 1, 0)  # 价格在均线下方时加分
        
        # 因子标准化（横截面Z-score）
        factor_cols = ['reversal_factor', 'size_factor', 'low_vol_factor', 'low_turnover_factor']
        for col in factor_cols:
            data[col + '_z'] = data.groupby('date')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        # 均线偏离直接使用原始值（已标准化在-1附近）
        data['ma_filter_z'] = data['ma_filter']
        
        # 等权合成综合因子
        data['composite_factor'] = (
            data['reversal_factor_z'] +
            data['size_factor_z'] +
            data['low_vol_factor_z'] +
            data['low_turnover_factor_z'] +
            data['ma_filter_z']
        ) / 5
        
        # 过滤条件：排除停牌、新股（上市不足60日）和ST股票（名称含ST）
        data = data[data['close'] > 0]  # 排除停牌
        data = data[data['date'] >= data.groupby('code')['date'].transform('min') + pd.Timedelta(days=60)]  # 上市满60日
        data = data[~data['name'].str.contains('ST|退市', na=False)]  # 排除ST
        
        # 每月最后一个交易日调仓
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        # 标记每月最后一个交易日
        last_trading_day = data.groupby(['year', 'month'])['date'].transform('max')
        data['is_rebalance_day'] = (data['date'] == last_trading_day)
        
        # 仅在调仓日生成信号
        rebalance_data = data[data['is_rebalance_day']].copy()
        
        # 每月选股：取综合因子排名前20的股票
        def select_stocks(group):
            group = group.nlargest(20, 'composite_factor')
            return group
        
        selected = rebalance_data.groupby('date', group_keys=False).apply(select_stocks)
        
        # 权重分配：等权，但单只不超过20%
        def assign_weights(group):
            n = len(group)
            weight = 1.0 / n
            # 如果超过20%限制则调整（等权情况下通常不会超过）
            weight = min(weight, 0.20)
            group['weight'] = weight
            return group
        
        selected = selected.groupby('date', group_keys=False).apply(assign_weights)
        
        # 构建输出DataFrame
        signals_df = selected[['code', 'weight', 'date']].reset_index(drop=True)
        
        return signals_df