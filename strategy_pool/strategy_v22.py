import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class AShareMultiFactorStrategy(BaseStrategy):
    name = "A股多因子选股策略"
    version = 1
    description = "基于反转、低波、低换手多因子等权合成的月度选股策略"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 数据排序，避免前向偏差
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 计算因子时使用shift避免前向偏差（用过去数据）
        # 1. 反转因子：过去20日涨跌幅（负值表示超跌，预期反转）
        data['reversal'] = -data.groupby('code')['close'].transform(lambda x: x.pct_change(20).shift(1))
        
        # 2. 低波动因子：过去20日波动率（负值表示低波动）
        data['low_vol'] = -data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std().shift(1))
        
        # 3. 低换手因子：过去20日平均换手率（负值表示低换手）
        data['low_turnover'] = -data.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean().shift(1))
        
        # 4. 均线偏离因子：当前价格偏离60日均线的程度（负值表示超跌）
        data['ma_deviation'] = -((data['close'] / data.groupby('code')['close'].transform(lambda x: x.rolling(60).mean().shift(1))) - 1)
        
        # 因子标准化（横截面z-score）
        factor_cols = ['reversal', 'low_vol', 'low_turnover', 'ma_deviation']
        for col in factor_cols:
            # 按日期分组标准化
            grouped = data.groupby('date')[col]
            data[col + '_z'] = (data[col] - grouped.transform('mean')) / grouped.transform('std')
        
        # 合成综合得分（等权）
        data['composite_score'] = data[[c + '_z' for c in factor_cols]].mean(axis=1)
        
        # 过滤掉缺失值
        data = data.dropna(subset=['composite_score'])
        
        # 月度调仓：每月最后一个交易日
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        # 找到每月最后一个交易日
        last_trading_days = data.groupby(['year', 'month'])['date'].transform('max')
        data = data[data['date'] == last_trading_days]
        
        # 每期选择得分最高的20只股票（等权）
        signals = []
        for date, group in data.groupby('date'):
            # 选择得分前20的股票
            top_stocks = group.nlargest(20, 'composite_score')
            # 等权分配
            weight = 1.0 / len(top_stocks)
            for _, row in top_stocks.iterrows():
                signals.append({
                    'code': row['code'],
                    'weight': weight,
                    'date': date
                })
        
        signals_df = pd.DataFrame(signals)
        return signals_df[['code', 'weight', 'date']]