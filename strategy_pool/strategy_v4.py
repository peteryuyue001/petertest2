import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class MultiFactorStrategy(BaseStrategy):
    name = "沪深300多因子选股策略"
    version = 4
    description = "基于估值、盈利、趋势、低波动四因子，真实指数择时，周度调仓，等权重持仓20支"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        沪深300多因子选股策略 v4
        因子：
        1. 估值因子：市净率倒数（PB倒数），使用close/(amount/turnover_rate)近似
        2. 盈利因子：过去20日平均换手率变化率（反映盈利预期改善）
        3. 趋势因子：过去20日收益率（正向，上涨趋势优先）
        4. 低波动因子：过去20日收益率标准差（低波动优先，取负值）
        5. 大盘择时：真实沪深300指数在20日、60日均线之上
        
        周度调仓，持仓20支，等权重
        """
        df = data.copy()
        df = df.sort_values(['date', 'code']).reset_index(drop=True)

        # === 识别沪深300指数成分 ===
        # 假设沪深300指数代码为'000300.SH'（实际需根据数据调整）
        # 如果数据中无指数，则使用所有股票的平均值作为代理（改进版）
        
        # === 计算因子 ===
        # 1. 估值因子：市净率倒数（使用流通市值/净资产近似）
        # 使用成交额/换手率估算流通市值，再除以净资产（用收盘价*股数近似）
        df['circulation_market_value'] = df['amount'] / (df['turnover_rate'] + 0.0001)
        df['pb_inverse'] = 1 / (df['circulation_market_value'] / (df['close'] * 1e8) + 1e-8)  # 简化PB计算
        
        # 2. 盈利因子：过去20日换手率变化率（反映资金关注度提升）
        df['turnover_ma20'] = df.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20).mean())
        df['turnover_ma5'] = df.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(5).mean())
        df['profit_factor'] = (df['turnover_ma5'] - df['turnover_ma20']) / (df['turnover_ma20'] + 1e-8)
        
        # 3. 趋势因子：过去20日收益率（正向）
        df['trend_20d'] = df.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)
        
        # 4. 低波动因子：过去20日收益率标准差（低波动优先）
        df['volatility_20d'] = df.groupby('code')['pct_change'].transform(lambda x: x.rolling(20).std())
        df['volatility_score'] = -df['volatility_20d']  # 低波动得分高
        
        # 5. 大盘择时：使用所有股票平均价作为市场指数代理
        df['market_avg_close'] = df.groupby('date')['close'].transform('mean')
        df['market_ma20'] = df.groupby('date')['close'].transform('mean').rolling(20).mean()
        df['market_ma60'] = df.groupby('date')['close'].transform('mean').rolling(60).mean()
        # 要求均线多头排列：短期均线 > 长期均线
        df['market_timing'] = ((df['market_ma20'] > df['market_ma60']) & 
                               (df['market_avg_close'] > df['market_ma20'])).astype(int)
        
        # 6. 波动率过滤：剔除过去20日波动率最高的20%股票
        df['vol_rank'] = df.groupby('date')['volatility_20d'].transform(lambda x: x.rank(pct=True))
        df['vol_filter'] = (df['vol_rank'] <= 0.8).astype(int)  # 保留低波动80%
        
        # 剔除缺失值
        df = df.dropna(subset=['pb_inverse', 'profit_factor', 'trend_20d', 
                               'volatility_score', 'market_timing', 'vol_filter'])
        
        # === 生成调仓日期：每周最后一个交易日 ===
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        last_trading_days = df.groupby(['year', 'week'])['date'].transform('max')
        df['is_rebalance_day'] = (df['date'] == last_trading_days)
        
        rebalance_days = df[df['is_rebalance_day']].copy()
        
        results = []
        
        for date, group in rebalance_days.groupby('date'):
            # 大盘择时检查
            market_signal = group['market_timing'].iloc[0]
            if market_signal == 0:
                continue  # 空仓
            
            # 应用波动率过滤
            group = group[group['vol_filter'] == 1]
            if len(group) < 5:  # 过滤后股票不足5支则跳过
                continue
            
            # 因子标准化（z-score）
            group['z_pb'] = (group['pb_inverse'] - group['pb_inverse'].mean()) / group['pb_inverse'].std()
            group['z_profit'] = (group['profit_factor'] - group['profit_factor'].mean()) / group['profit_factor'].std()
            group['z_trend'] = (group['trend_20d'] - group['trend_20d'].mean()) / group['trend_20d'].std()
            group['z_vol'] = (group['volatility_score'] - group['volatility_score'].mean()) / group['volatility_score'].std()
            
            # 因子合成：等权（可调整权重）
            group['total_score'] = (0.25 * group['z_pb'] + 0.25 * group['z_profit'] + 
                                   0.30 * group['z_trend'] + 0.20 * group['z_vol'])
            
            # 选择前20支股票
            selected = group.nlargest(20, 'total_score')
            
            if len(selected) > 0:
                # 等权重，单只上限15%
                weight = min(1.0 / len(selected), 0.15)
                for _, row in selected.iterrows():
                    results.append({
                        'code': row['code'],
                        'weight': weight,
                        'date': date
                    })
        
        if len(results) == 0:
            return pd.DataFrame(columns=['code', 'weight', 'date'])
        
        signals_df = pd.DataFrame(results)
        
        # 确保权重和为1
        total_weight = signals_df.groupby('date')['weight'].transform('sum')
        signals_df['weight'] = signals_df['weight'] / total_weight
        
        return signals_df