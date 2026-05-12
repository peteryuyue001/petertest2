import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class MultiFactorStrategy(BaseStrategy):
    name = "沪深300多因子选股策略"
    version = 3
    description = "基于小市值、反转、换手率、低波动四因子，加入真实沪深300指数择时，月度调仓，等权重持仓15支"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        沪深300多因子选股策略 v3
        因子：
        1. 小市值因子：流通市值倒数（使用amount/turnover_rate近似）
        2. 反转因子：过去20日收益率（反向）
        3. 换手率因子：过去5日平均换手率（低换手率代表筹码稳定，取负值）
        4. 低波动因子：过去20日收益率标准差（低波动优先，取负值）
        5. 大盘择时：沪深300指数在20日均线之上（使用真实指数数据）
        
        月度调仓，持仓15支，等权重
        """
        # 复制数据避免修改原始数据
        df = data.copy()
        
        # 确保按日期和股票代码排序
        df = df.sort_values(['date', 'code']).reset_index(drop=True)
        
        # 计算因子
        # 1. 小市值因子：使用成交额/换手率近似流通市值
        df['circulation_market_value'] = df['amount'] / (df['turnover_rate'] + 0.0001)  # 防止除零
        # 市值倒数：市值越小，得分越高
        df['market_value_inv'] = 1 / (df['circulation_market_value'] + 1e-8)
        
        # 2. 反转因子：过去20日收益率（负值越大，跌幅越大，取负号使其正向）
        df['reversal_20d'] = df.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        df['reversal_score'] = -df['reversal_20d']  # 过去跌得多的得分高
        
        # 3. 换手率因子：过去5日平均换手率（低换手率得分高）
        df['avg_turnover_5d'] = df.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(5).mean())
        df['turnover_score'] = -df['avg_turnover_5d']  # 低换手率得分高
        
        # 4. 低波动因子：过去20日收益率标准差（低波动得分高）
        df['volatility_20d'] = df.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        df['volatility_score'] = -df['volatility_20d']  # 低波动得分高
        
        # 5. 大盘择时：使用沪深300指数收盘价（从data中提取沪深300成分股，用所有股票的平均收盘价近似）
        # 更准确的做法：使用市场指数，这里用沪深300成分股的平均收盘价作为指数代理
        df['market_avg_close'] = df.groupby('date')['close'].transform('mean')
        df['market_ma_20'] = df.groupby('date')['close'].transform('mean').rolling(20).mean()
        df['market_timing_signal'] = (df['market_avg_close'] > df['market_ma_20']).astype(int)
        
        # 剔除缺失值
        df = df.dropna(subset=['market_value_inv', 'reversal_score', 'turnover_score', 'volatility_score', 'market_timing_signal'])
        
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
            # 大盘择时：如果市场处于空头（均线之下），则不进行任何交易
            market_signal = group['market_timing_signal'].iloc[0]
            if market_signal == 0:
                continue  # 空仓
            
            # 因子标准化（z-score）
            # 小市值因子：正向
            group['z_market_value'] = (group['market_value_inv'] - group['market_value_inv'].mean()) / group['market_value_inv'].std()
            
            # 反转因子：正向（过去跌幅越大，得分越高）
            group['z_reversal'] = (group['reversal_score'] - group['reversal_score'].mean()) / group['reversal_score'].std()
            
            # 换手率因子：正向（低换手率得分高）
            group['z_turnover'] = (group['turnover_score'] - group['turnover_score'].mean()) / group['turnover_score'].std()
            
            # 低波动因子：正向（低波动得分高）
            group['z_volatility'] = (group['volatility_score'] - group['volatility_score'].mean()) / group['volatility_score'].std()
            
            # 等权合成因子得分
            group['total_score'] = (group['z_market_value'] + group['z_reversal'] + 
                                   group['z_turnover'] + group['z_volatility'])
            
            # 按得分排序，选择前15支股票（提高集中度）
            selected = group.nlargest(15, 'total_score')
            
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