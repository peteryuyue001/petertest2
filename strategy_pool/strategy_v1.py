import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class MultiFactorStrategy(BaseStrategy):
    name = "沪深300多因子选股策略"
    version = 1
    description = "基于20日动量、波动率过滤、换手率因子的多因子选股策略，月度调仓，等权重持仓20支"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        沪深300多因子选股策略
        因子：
        1. 20日动量因子：过去20日收益率
        2. 波动率过滤：剔除过去20日波动率最高的30%股票
        3. 换手率因子：过去20日平均换手率（反向）
        
        月度调仓，持仓20支，等权重
        """
        # 复制数据避免修改原始数据
        df = data.copy()
        
        # 确保按日期和股票代码排序
        df = df.sort_values(['date', 'code']).reset_index(drop=True)
        
        # 计算因子
        # 1. 20日动量因子：过去20日收益率
        df['momentum_20d'] = df.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        
        # 2. 波动率因子：过去20日收益率标准差
        df['volatility_20d'] = df.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        
        # 3. 换手率因子：过去20日平均换手率
        df['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        
        # 剔除缺失值
        df = df.dropna(subset=['momentum_20d', 'volatility_20d', 'turnover_20d'])
        
        # 获取每个交易日的股票池（沪深300成分股）
        # 假设data中只包含沪深300成分股数据
        
        # 生成调仓日期：每月最后一个交易日
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        # 找到每月最后一个交易日
        last_trading_days = df.groupby(['year', 'month'])['date'].transform('max')
        df['is_rebalance_day'] = (df['date'] == last_trading_days)
        
        # 只保留调仓日数据
        rebalance_days = df[df['is_rebalance_day']].copy()
        
        # 在每个调仓日进行选股
        results = []
        
        for date, group in rebalance_days.groupby('date'):
            # 波动率过滤：剔除波动率最高的30%股票
            volatility_threshold = group['volatility_20d'].quantile(0.7)
            filtered = group[group['volatility_20d'] <= volatility_threshold].copy()
            
            if len(filtered) == 0:
                continue
            
            # 因子标准化（z-score）
            # 动量因子：正向
            filtered['momentum_z'] = (filtered['momentum_20d'] - filtered['momentum_20d'].mean()) / filtered['momentum_20d'].std()
            
            # 换手率因子：反向（低换手率得分高）
            filtered['turnover_z'] = (filtered['turnover_20d'].mean() - filtered['turnover_20d']) / filtered['turnover_20d'].std()
            
            # 等权合成因子得分
            filtered['total_score'] = filtered['momentum_z'] + filtered['turnover_z']
            
            # 按得分排序，选择前20支股票
            selected = filtered.nlargest(20, 'total_score')
            
            # 生成信号
            if len(selected) > 0:
                # 等权重
                weight = 1.0 / len(selected)
                for _, row in selected.iterrows():
                    results.append({
                        'code': row['code'],
                        'weight': weight,
                        'date': date
                    })
        
        # 如果没有选到任何股票，返回空DataFrame
        if len(results) == 0:
            return pd.DataFrame(columns=['code', 'weight', 'date'])
        
        # 构建结果DataFrame
        signals_df = pd.DataFrame(results)
        
        return signals_df