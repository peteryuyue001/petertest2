import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class CSI300MultiFactorStrategy(BaseStrategy):
    name = "沪深300多因子选股策略"
    version = 1
    description = "基于20日动量、波动率过滤、换手率因子的沪深300多因子选股策略，月度调仓，等权持仓20支"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # ===== 1. 数据预处理 =====
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date']).reset_index(drop=True)

        # ===== 2. 因子计算 =====
        # 2.1 20日动量因子
        df['momentum_20'] = df.groupby('code')['close'].transform(
            lambda x: x / x.shift(20) - 1
        )
        
        # 2.2 波动率因子（20日标准差）
        df['volatility_20'] = df.groupby('code')['pct_change'].transform(
            lambda x: x.rolling(20).std()
        )
        
        # 2.3 换手率因子（20日均值）
        df['turnover_20'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(20).mean()
        )

        # ===== 3. 过滤条件 =====
        # 3.1 波动率过滤：剔除波动率最高的20%
        vol_threshold = df.groupby('date')['volatility_20'].transform(
            lambda x: x.quantile(0.8)
        )
        df['vol_filter'] = df['volatility_20'] < vol_threshold
        
        # 3.2 换手率过滤：剔除换手率最低的20%
        turnover_threshold = df.groupby('date')['turnover_20'].transform(
            lambda x: x.quantile(0.2)
        )
        df['turnover_filter'] = df['turnover_20'] > turnover_threshold

        # ===== 4. 因子合成 =====
        # 对动量因子进行截面标准化（Z-score）
        df['momentum_z'] = df.groupby('date')['momentum_20'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # 对换手率因子进行截面标准化（Z-score）
        df['turnover_z'] = df.groupby('date')['turnover_20'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # 等权合成综合因子（动量正向，换手率正向）
        df['composite_factor'] = df['momentum_z'] + df['turnover_z']

        # ===== 5. 选股逻辑 =====
        # 仅保留每月最后一个交易日
        df['year_month'] = df['date'].dt.to_period('M')
        df['month_end'] = df.groupby(['code', 'year_month'])['date'].transform('max')
        df_monthly = df[df['date'] == df['month_end']].copy()

        # 应用过滤条件
        df_monthly = df_monthly[
            df_monthly['vol_filter'] & 
            df_monthly['turnover_filter'] & 
            df_monthly['composite_factor'].notna()
        ]

        # 按综合因子排序，选前20支
        df_monthly['rank'] = df_monthly.groupby('date')['composite_factor'].rank(
            ascending=False, method='first'
        )
        df_selected = df_monthly[df_monthly['rank'] <= 20].copy()

        # ===== 6. 生成信号 =====
        # 等权分配
        df_selected['weight'] = 1.0 / 20
        
        # 构建输出
        signals_df = df_selected[['code', 'weight', 'date']].copy()
        signals_df = signals_df.reset_index(drop=True)
        
        # 确保权重不超过20%
        signals_df['weight'] = signals_df['weight'].clip(upper=0.2)
        
        return signals_df