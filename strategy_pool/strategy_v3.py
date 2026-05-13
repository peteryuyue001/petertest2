import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class HS300MultiFactorStrategy(BaseStrategy):
    name = "沪深300多因子选股策略"
    version = 1
    description = "基于20日动量、波动率过滤、换手率因子的月度调仓策略，等权重持有20支股票"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # ===== 数据预处理 =====
        # 按股票代码和日期排序，确保时间序列正确
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 确保date列为字符串类型
        if data['date'].dtype != 'object':
            data['date'] = data['date'].astype(str)
        
        # ===== 因子计算 =====
        # 1. 20日动量因子（反转效应：过去涨幅过高的股票未来可能回调）
        data['momentum_20'] = data.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        
        # 2. 波动率因子（20日波动率，用于过滤高波动股票）
        data['volatility_20'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        
        # 3. 换手率因子（20日平均换手率，低换手率表示筹码稳定）
        data['turnover_20'] = data.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        
        # ===== 因子处理 =====
        # 过滤掉缺失值
        data = data.dropna(subset=['momentum_20', 'volatility_20', 'turnover_20'])
        
        # 因子标准化（Z-score）
        for factor in ['momentum_20', 'volatility_20', 'turnover_20']:
            mean = data.groupby('date')[factor].transform('mean')
            std = data.groupby('date')[factor].transform('std')
            data[f'{factor}_zscore'] = (data[factor] - mean) / std.replace(0, np.nan)
        
        # ===== 综合评分 =====
        # 动量因子取负值（反转效应：过去跌得多的股票未来可能涨）
        # 波动率因子取负值（低波动股票更稳定）
        # 换手率因子取负值（低换手率代表筹码锁定）
        data['score'] = -data['momentum_20_zscore'] - data['volatility_20_zscore'] - data['turnover_20_zscore']
        
        # ===== 选股逻辑 =====
        # 使用字符串切片提取年月，避免.str访问器问题
        data['year_month'] = data['date'].str[:7]
        
        # 标记每月最后一个交易日
        data['month_end'] = data.groupby(['code', 'year_month'])['date'].transform('max')
        data['is_month_end'] = data['date'] == data['month_end']
        
        # 只保留月末交易日的数据
        month_end_data = data[data['is_month_end']].copy()
        
        # 在每个月末，按评分排序选前20支股票
        def select_stocks(group):
            # 按评分降序排列
            group = group.sort_values('score', ascending=False)
            # 选前20支
            selected = group.head(20)
            # 等权重分配
            selected['weight'] = 1.0 / len(selected)
            return selected
        
        selected = month_end_data.groupby('date', group_keys=False).apply(select_stocks)
        
        # ===== 生成信号 =====
        # 构建返回结果
        signals_df = selected[['code', 'weight', 'date']].reset_index(drop=True)
        
        # 确保权重在0~1之间
        signals_df['weight'] = signals_df['weight'].clip(0, 0.2)  # 单支持仓不超过20%
        
        return signals_df