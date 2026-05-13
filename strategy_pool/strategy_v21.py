import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class StrategyV20(BaseStrategy):
    name = "多因子复合选股策略v20"
    version = 20
    description = "基于反转、小市值、低波动、低换手率的多因子等权复合选股策略，月度调仓，持仓10-20只"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        策略逻辑：
        1. 计算四个因子：反转因子、小市值因子、低波动因子、低换手因子
        2. 因子标准化后等权合成综合得分
        3. 选取得分最高的15只股票，等权重配置
        4. 每月最后一个交易日调仓
        """
        # 复制数据避免修改原始数据
        df = data.copy()
        
        # 确保日期排序
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 计算因子
        # 1. 反转因子：过去20日收益率（负值越大表示跌幅越大，预期反转）
        df['momentum_20'] = df.groupby('code')['close'].pct_change(20)
        df['reversal_factor'] = -df['momentum_20']  # 反转因子，跌幅大的得分高
        
        # 2. 小市值因子：使用流通市值（通过价格*流通股本近似），这里用成交额/换手率近似
        # 实际可用amount/volume估算，但更精确用close * 流通股，这里用turnover_rate辅助
        df['market_cap_factor'] = -df['close'] * df['volume'] / (df['turnover_rate'] + 0.0001)
        # 标准化为排名
        df['market_cap_rank'] = df.groupby('date')['market_cap_factor'].rank(pct=True)
        df['small_cap_factor'] = 1 - df['market_cap_rank']  # 市值越小得分越高
        
        # 3. 低波动因子：过去20日收益率标准差
        df['volatility_20'] = df.groupby('code')['pct_change'].rolling(20).std().reset_index(level=0, drop=True)
        df['low_vol_factor'] = -df['volatility_20']  # 波动率越低得分越高
        
        # 4. 低换手因子：过去20日平均换手率倒数
        df['turnover_20'] = df.groupby('code')['turnover_rate'].rolling(20).mean().reset_index(level=0, drop=True)
        df['low_turnover_factor'] = -df['turnover_20']  # 换手率越低得分越高
        
        # 因子标准化（横截面z-score）
        factor_cols = ['reversal_factor', 'small_cap_factor', 'low_vol_factor', 'low_turnover_factor']
        for col in factor_cols:
            df[col + '_zscore'] = df.groupby('date')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        
        # 等权合成综合得分
        zscore_cols = [col + '_zscore' for col in factor_cols]
        df['composite_score'] = df[zscore_cols].mean(axis=1)
        
        # 过滤条件
        # 排除ST、新股（上市不足60日）、停牌（换手率为0）等
        df = df[df['turnover_rate'] > 0]  # 排除停牌
        df = df[df['close'] > 0]  # 排除异常价格
        
        # 计算上市天数（用数据中最早日期近似）
        earliest_date = df['date'].min()
        df['days_listed'] = (df['date'] - earliest_date).dt.days
        df = df[df['days_listed'] >= 60]  # 排除次新股
        
        # 每月最后一个交易日
        df['year_month'] = df['date'].dt.to_period('M')
        df['is_month_end'] = df.groupby('year_month')['date'].transform(lambda x: x == x.max())
        
        # 只在调仓日选股
        signal_dates = df[df['is_month_end']]['date'].unique()
        
        signals_list = []
        
        for trade_date in signal_dates:
            # 获取调仓日数据
            day_data = df[df['date'] == trade_date].copy()
            
            # 按综合得分排序，选取得分最高的15只
            day_data = day_data.sort_values('composite_score', ascending=False)
            selected = day_data.head(15)
            
            # 等权重配置
            weight = 1.0 / len(selected) if len(selected) > 0 else 0
            
            for _, row in selected.iterrows():
                signals_list.append({
                    'code': row['code'],
                    'weight': weight,
                    'date': trade_date
                })
        
        # 创建信号DataFrame
        signals_df = pd.DataFrame(signals_list)
        
        # 确保列顺序正确
        signals_df = signals_df[['code', 'weight', 'date']]
        
        return signals_df