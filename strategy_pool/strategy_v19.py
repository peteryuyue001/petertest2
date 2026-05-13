import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class CSI300MultiFactorV2(BaseStrategy):
    name = "沪深300多因子选股策略-v2"
    version = 2
    description = "基于20日反转、低波动、低换手和小市值因子的沪深300选股策略，月度调仓，等权持有30支股票"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 确保数据按日期和代码排序
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        
        # 计算因子
        # 1. 20日反转因子（过去20日跌幅越大，得分越高）
        data['reversal'] = data.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        # 反转因子取负值，使得跌幅大的股票得分高
        data['reversal'] = -data['reversal']
        
        # 2. 波动率因子（过去20日收益率标准差，用于过滤高波动股票）
        data['volatility'] = data.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        # 低波动偏好，取负值
        data['volatility'] = -data['volatility']
        
        # 3. 换手率因子（过去20日平均换手率）
        data['turnover_avg'] = data.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        # 低换手偏好，取负值
        data['turnover_avg'] = -data['turnover_avg']
        
        # 4. 小市值因子（使用成交额作为市值的近似代理，成交额越低，市值越小，得分越高）
        data['amount_avg'] = data.groupby('code')['amount'].transform(lambda x: x.rolling(20).mean())
        # 成交额越小越好，取负值
        data['amount_avg'] = -data['amount_avg']
        
        # 数据清洗：剔除缺失值
        data = data.dropna(subset=['reversal', 'volatility', 'turnover_avg', 'amount_avg'])
        
        # 因子处理：去极值、标准化
        def winsorize_and_standardize(series):
            # 去极值（3倍标准差）
            mean = series.mean()
            std = series.std()
            series = series.clip(mean - 3*std, mean + 3*std)
            # 标准化
            return (series - series.mean()) / series.std()
        
        # 按日期分组处理因子
        data['reversal_z'] = data.groupby('date')['reversal'].transform(winsorize_and_standardize)
        data['volatility_z'] = data.groupby('date')['volatility'].transform(winsorize_and_standardize)
        data['turnover_z'] = data.groupby('date')['turnover_avg'].transform(winsorize_and_standardize)
        data['amount_z'] = data.groupby('date')['amount_avg'].transform(winsorize_and_standardize)
        
        # 等权合成综合因子
        data['combined_score'] = (data['reversal_z'] + data['volatility_z'] + data['turnover_z'] + data['amount_z']) / 4
        
        # 月度调仓：获取每月最后一个交易日
        data['year_month'] = data['date'].astype(str).str[:7]
        monthly_dates = data.groupby('year_month')['date'].max().reset_index()
        monthly_dates = monthly_dates[['date']].dropna()
        
        # 生成调仓信号
        signals_list = []
        
        for rebalance_date in monthly_dates['date'].unique():
            # 获取该调仓日期的数据（使用前一个交易日的数据避免前向偏差）
            rebalance_data = data[data['date'] == rebalance_date].copy()
            
            if rebalance_data.empty:
                continue
            
            # 选股：选择综合得分最高的30支股票
            rebalance_data = rebalance_data.sort_values('combined_score', ascending=False)
            top_stocks = rebalance_data.head(30)
            
            # 等权分配权重
            weight = 1.0 / len(top_stocks)
            top_stocks['weight'] = weight
            
            # 构建信号DataFrame
            signals = top_stocks[['code', 'weight']].copy()
            signals['date'] = rebalance_date
            signals_list.append(signals)
        
        # 合并所有调仓信号
        if signals_list:
            signals_df = pd.concat(signals_list, ignore_index=True)
        else:
            signals_df = pd.DataFrame(columns=['code', 'weight', 'date'])
        
        return signals_df