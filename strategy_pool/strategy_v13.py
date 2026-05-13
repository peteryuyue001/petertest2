import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class MultiFactorHS300StrategyV2(BaseStrategy):
    name = "沪深300多因子选股策略V2"
    version = 2
    description = "基于20日反转、小市值、低换手因子的沪深300多因子选股策略，月度调仓，持仓15支等权重"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 复制数据避免修改原数据
        df = data.copy()
        
        # 确保日期排序
        df = df.sort_values(['code', 'date'])
        
        # ===== 因子计算 =====
        # 1. 20日反转因子（过去20日收益率为负，预期未来上涨）
        df['reversal'] = df.groupby('code')['pct_change'].transform(
            lambda x: x.rolling(20).sum()
        )
        # 反转因子取负值，使过去跌幅大的股票获得高评分
        df['reversal'] = -df['reversal']
        
        # 2. 小市值因子（市值越小，预期收益越高）
        # 使用收盘价 * 流通股本（近似市值），实际可用amount/volume估算
        # 这里用close * volume作为市值代理，注意单位
        df['market_cap'] = df['close'] * df['volume'] / 1e8  # 以亿为单位
        # 小市值因子直接取市值负值
        df['small_cap'] = -df['market_cap']
        
        # 3. 低换手率因子（换手率越低，筹码越稳定）
        df['low_turnover'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(20).mean()
        )
        # 低换手率因子取负值
        df['low_turnover'] = -df['low_turnover']
        
        # ===== 因子处理 =====
        # 去除缺失值和无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['reversal', 'small_cap', 'low_turnover'])
        
        # 因子标准化（横截面Z-score）
        def zscore(group):
            return (group - group.mean()) / group.std()
        
        df['reversal_z'] = df.groupby('date')['reversal'].transform(zscore)
        df['small_cap_z'] = df.groupby('date')['small_cap'].transform(zscore)
        df['low_turnover_z'] = df.groupby('date')['low_turnover'].transform(zscore)
        
        # 等权合成因子
        df['composite_factor'] = (df['reversal_z'] + df['small_cap_z'] + df['low_turnover_z']) / 3
        
        # ===== 大盘择时过滤 =====
        # 计算沪深300指数20日均线，判断市场趋势
        # 假设data中包含所有股票，这里用所有股票的平均价格作为大盘代理
        # 更准确的做法是直接使用沪深300指数数据，但限于数据可用性，这里用全部股票等权平均
        df_market = df.groupby('date')['close'].mean().reset_index()
        df_market.columns = ['date', 'market_close']
        df_market['ma20'] = df_market['market_close'].rolling(20).mean()
        df_market['market_trend'] = np.where(
            df_market['market_close'] > df_market['ma20'], 1, 0
        )
        # 将市场趋势合并回原数据
        df = df.merge(df_market[['date', 'market_trend']], on='date', how='left')
        
        # ===== 选股逻辑 =====
        # 月度调仓：每月最后一个交易日调仓
        df['year_month'] = df['date'].dt.to_period('M')
        df['is_month_end'] = df.groupby('year_month')['date'].transform(lambda x: x == x.max())
        
        # 只保留调仓日数据
        signal_df = df[df['is_month_end']].copy()
        
        # 应用大盘择时：只在市场趋势向上时选股，否则空仓
        # 市场趋势向下时，返回空信号
        if signal_df['market_trend'].iloc[0] == 0:
            # 返回空持仓
            result = pd.DataFrame(columns=['code', 'weight', 'date'])
            return result
        
        # 按调仓日分组，选取复合因子排名前15的股票
        signal_df['rank'] = signal_df.groupby('date')['composite_factor'].rank(ascending=False)
        signal_df = signal_df[signal_df['rank'] <= 15]
        
        # ===== 构建输出 =====
        # 等权重分配
        signal_df['weight'] = 1.0 / 15
        
        # 选择需要的列并重命名
        result = signal_df[['code', 'weight', 'date']].copy()
        result = result.reset_index(drop=True)
        
        # 确保权重精度
        result['weight'] = result['weight'].round(4)
        
        return result