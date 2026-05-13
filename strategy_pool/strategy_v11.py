import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class ImprovedHS300Strategy(BaseStrategy):
    name = "沪深300多因子选股策略（改进版）"
    version = 2
    description = "反转、低波动、低换手、小市值、价格位置五因子，周度调仓，持仓15支等权重，含大盘择时"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(['code', 'date'])
        
        # ===== 因子计算 =====
        # 1. 反转因子（-20日动量）
        df['momentum_20'] = df.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        df['reversal'] = -df['momentum_20']  # 反转因子
        
        # 2. 波动率因子（20日收益率标准差）
        df['volatility'] = df.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        
        # 3. 换手率因子（20日平均换手率）
        df['turnover'] = df.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        
        # 4. 小市值因子（流通市值倒数，用成交量*价格近似）
        df['market_value'] = df['close'] * df['volume']  # 近似市值
        df['small_cap'] = 1 / df['market_value']
        
        # 5. 价格位置因子（当前价/20日均价）
        df['ma20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
        df['price_position'] = df['close'] / df['ma20']
        
        # ===== 大盘择时 =====
        # 计算沪深300指数20日均线（使用所有股票平均价近似）
        index_data = df.groupby('date')['close'].mean().reset_index()
        index_data.columns = ['date', 'index_close']
        index_data['index_ma20'] = index_data['index_close'].rolling(20).mean()
        index_data['market_timing'] = np.where(index_data['index_close'] > index_data['index_ma20'], 1, 0)
        
        df = df.merge(index_data[['date', 'market_timing']], on='date', how='left')
        
        # ===== 因子处理 =====
        df = df.replace([np.inf, -np.inf], np.nan)
        # 保留足够历史数据
        df = df.dropna(subset=['reversal', 'volatility', 'turnover', 'small_cap', 'price_position'])
        
        # 行业中性化（使用简单的一级行业分类：金融、制造、科技、消费、其他）
        # 这里用行业代码模拟（实际需真实行业数据）
        def assign_sector(code):
            if code.startswith('60') or code.startswith('00'):
                return '金融'
            elif code.startswith('30'):
                return '科技'
            elif code.startswith('002'):
                return '制造'
            elif code.startswith('300'):
                return '消费'
            else:
                return '其他'
        df['sector'] = df['code'].apply(assign_sector)
        
        # 因子标准化（行业内Z-score）
        def sector_zscore(group):
            return (group - group.mean()) / group.std()
        
        for factor in ['reversal', 'volatility', 'turnover', 'small_cap', 'price_position']:
            df[f'{factor}_z'] = df.groupby(['date', 'sector'])[factor].transform(sector_zscore)
        
        # 因子方向：反转正向、低波动正向、低换手正向、小市值正向、价格位置正向（均值回复）
        df['volatility_z'] = -df['volatility_z']  # 低波动
        df['turnover_z'] = -df['turnover_z']      # 低换手
        
        # 等权合成因子
        df['composite_factor'] = (df['reversal_z'] + df['volatility_z'] + 
                                  df['turnover_z'] + df['small_cap_z'] + 
                                  df['price_position_z']) / 5
        
        # ===== 选股逻辑 =====
        # 周度调仓：每周最后一个交易日
        df['week'] = df['date'].dt.isocalendar().week
        df['year_week'] = df['date'].dt.year.astype(str) + '-' + df['week'].astype(str)
        df['is_week_end'] = df.groupby('year_week')['date'].transform(lambda x: x == x.max())
        
        # 只保留调仓日且市场择时信号为1
        signal_df = df[(df['is_week_end']) & (df['market_timing'] == 1)].copy()
        if signal_df.empty:
            # 市场信号为空时，返回空持仓
            return pd.DataFrame(columns=['code', 'weight', 'date'])
        
        # 按调仓日分组，选取复合因子排名前15的股票
        signal_df['rank'] = signal_df.groupby('date')['composite_factor'].rank(ascending=False)
        signal_df = signal_df[signal_df['rank'] <= 15]
        
        # ===== 构建输出 =====
        signal_df['weight'] = 1.0 / 15
        
        result = signal_df[['code', 'weight', 'date']].copy()
        result = result.reset_index(drop=True)
        result['weight'] = result['weight'].round(4)
        
        return result