import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class MultiFactorHS300StrategyV2(BaseStrategy):
    name = "沪深300多因子选股策略v2"
    version = 2
    description = "基于反转、高换手、小市值因子的沪深300选股策略，月度调仓，持仓20支等权重"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 复制数据避免修改原数据
        df = data.copy()
        
        # 确保日期排序
        df = df.sort_values(['code', 'date'])
        
        # ===== 因子计算 =====
        # 1. 20日反转因子（过去20日累计收益率，取负值表示反转）
        df['reversal'] = df.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        df['reversal'] = -df['reversal']  # 反转：过去跌得多，未来可能涨
        
        # 2. 20日平均换手率因子（高换手代表活跃）
        df['turnover'] = df.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        
        # 3. 小市值因子（使用流通市值近似，这里用成交额/换手率估算，或直接用流通市值列）
        # 注意：data中无直接市值列，使用收盘价 * 成交量 * 一个比例因子来近似相对大小
        # 更准确的做法是用amount/turnover_rate估算流通市值（如果turnover_rate是百分比）
        df['market_value_est'] = df['amount'] / (df['turnover_rate'] + 1e-8)  # 避免除零
        df['market_value_est'] = df.groupby('code')['market_value_est'].transform(lambda x: x.rolling(20).mean())
        
        # ===== 因子处理 =====
        # 去除缺失值和无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['reversal', 'turnover', 'market_value_est'])
        
        # 因子标准化（横截面Z-score）
        def zscore(group):
            return (group - group.mean()) / group.std()
        
        df['reversal_z'] = df.groupby('date')['reversal'].transform(zscore)
        df['turnover_z'] = df.groupby('date')['turnover'].transform(zscore)
        df['market_value_z'] = df.groupby('date')['market_value_est'].transform(zscore)
        
        # 因子方向：反转（正）、高换手（正）、小市值（负的市值因子，即-市值）
        df['market_value_z'] = -df['market_value_z']  # 小市值因子
        
        # 等权合成因子
        df['composite_factor'] = (df['reversal_z'] + df['turnover_z'] + df['market_value_z']) / 3
        
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