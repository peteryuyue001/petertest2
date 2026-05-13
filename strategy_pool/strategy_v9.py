import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class MultiFactorHS300Strategy(BaseStrategy):
    name = "沪深300多因子选股策略"
    version = 1
    description = "基于20日动量、波动率过滤、换手率因子的沪深300多因子选股策略，月度调仓，持仓20支等权重"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 复制数据避免修改原数据
        df = data.copy()
        
        # 确保日期排序
        df = df.sort_values(['code', 'date'])
        
        # ===== 因子计算 =====
        # 1. 20日动量因子
        df['momentum'] = df.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        
        # 2. 波动率因子（20日收益率标准差）
        df['volatility'] = df.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        
        # 3. 换手率因子（20日平均换手率）
        df['turnover'] = df.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        
        # ===== 因子处理 =====
        # 去除缺失值和无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['momentum', 'volatility', 'turnover'])
        
        # 因子标准化（横截面Z-score）
        def zscore(group):
            return (group - group.mean()) / group.std()
        
        df['momentum_z'] = df.groupby('date')['momentum'].transform(zscore)
        df['volatility_z'] = df.groupby('date')['volatility'].transform(zscore)
        df['turnover_z'] = df.groupby('date')['turnover'].transform(zscore)
        
        # 因子方向调整：动量正向、波动率负向（低波动）、换手率负向（低换手）
        df['volatility_z'] = -df['volatility_z']
        df['turnover_z'] = -df['turnover_z']
        
        # 等权合成因子
        df['composite_factor'] = (df['momentum_z'] + df['volatility_z'] + df['turnover_z']) / 3
        
        # ===== 选股逻辑 =====
        # 月度调仓：每月最后一个交易日调仓
        df['year_month'] = df['date'].dt.to_period('M')
        df['is_month_end'] = df.groupby('year_month')['date'].transform(lambda x: x == x.max())
        
        # 只保留调仓日数据
        signal_df = df[df['is_month_end']].copy()
        
        # 按调仓日分组，选取复合因子排名前20的股票
        signal_df['rank'] = signal_df.groupby('date')['composite_factor'].rank(ascending=False)
        signal_df = signal_df[signal_df['rank'] <= 20]
        
        # ===== 构建输出 =====
        # 等权重分配
        signal_df['weight'] = 1.0 / 20
        
        # 选择需要的列并重命名
        result = signal_df[['code', 'weight', 'date']].copy()
        result = result.reset_index(drop=True)
        
        # 确保权重精度
        result['weight'] = result['weight'].round(4)
        
        return result