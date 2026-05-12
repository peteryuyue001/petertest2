import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class MultiFactorStrategy(BaseStrategy):
    name = "沪深300多因子选股策略"
    version = 2
    description = "基于小市值、反转因子，加入均线择时过滤，月度调仓，等权重持仓30支"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        沪深300多因子选股策略 v2
        因子：
        1. 小市值因子：流通市值（使用收盘价 * 流通股本近似）
        2. 反转因子：过去5日收益率（反向）
        3. 大盘择时：沪深300指数收盘价在20日均线之上
        
        月度调仓，持仓30支，等权重
        """
        # 复制数据避免修改原始数据
        df = data.copy()
        
        # 确保按日期和股票代码排序
        df = df.sort_values(['date', 'code']).reset_index(drop=True)
        
        # 计算因子
        # 1. 小市值因子：使用收盘价 * 流通股本（这里用成交额/换手率近似流通市值，更准确的是amount/turnover_rate）
        # 注意：turnover_rate可能为0，需要处理
        df['circulation_market_value'] = df['amount'] / (df['turnover_rate'] + 0.0001)  # 防止除零
        
        # 2. 反转因子：过去5日收益率（负值越大，跌幅越大，我们想要买入跌幅大的，所以取负号使其正向）
        df['reversal_5d'] = df.groupby('code')['close'].transform(lambda x: x / x.shift(5) - 1)
        # 反转因子：我们希望买入过去跌的，所以取负值
        df['reversal_score'] = -df['reversal_5d']
        
        # 3. 大盘择时：计算沪深300指数（这里用所有股票的平均收盘价作为指数近似，更标准的做法应该是使用市场指数数据）
        # 由于data中可能没有单独的指数数据，我们使用所有股票的平均价格来模拟
        df['market_avg_close'] = df.groupby('date')['close'].transform('mean')
        df['market_ma_20'] = df['market_avg_close'].rolling(20).mean()
        # 择时信号：当前价格在20日均线之上
        df['market_timing_signal'] = (df['market_avg_close'] > df['market_ma_20']).astype(int)
        
        # 剔除缺失值
        df = df.dropna(subset=['circulation_market_value', 'reversal_score', 'market_timing_signal'])
        
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
            # 我们取当前日期所有股票的第一条记录来获取择时信号
            market_signal = group['market_timing_signal'].iloc[0]
            if market_signal == 0:
                continue  # 空仓
            
            # 因子标准化（z-score）
            # 小市值因子：正向（市值越小，得分越高）
            # 注意：市值越小，值越小，所以取负号
            group['market_value_z'] = - (group['circulation_market_value'] - group['circulation_market_value'].mean()) / group['circulation_market_value'].std()
            
            # 反转因子：正向（过去跌幅越大，得分越高）
            group['reversal_z'] = (group['reversal_score'] - group['reversal_score'].mean()) / group['reversal_score'].std()
            
            # 等权合成因子得分
            group['total_score'] = group['market_value_z'] + group['reversal_z']
            
            # 按得分排序，选择前30支股票
            selected = group.nlargest(30, 'total_score')
            
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