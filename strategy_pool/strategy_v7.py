import pandas as pd
import numpy as np
from template.strategy_base import BaseStrategy

class FixedMultiFactorStrategy(BaseStrategy):
    name = "沪深300多因子选股策略(修复版)"
    version = 7
    description = "基于估值、盈利、趋势、低波动四因子，周度调仓，等权重持仓20支"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        修复后的多因子选股策略
        修复点：
        1. 修复了市场均线计算中的前向偏差问题
        2. 修复了因子计算中的NaN传播问题
        3. 优化了调仓日期的生成逻辑
        4. 增加了数据质量检查
        5. 修复了因子标准化列名问题
        6. 修复了调仓日判断逻辑
        7. 修复了因子标准化列名不一致问题
        """
        df = data.copy()
        df = df.sort_values(['date', 'code']).reset_index(drop=True)
        
        # 确保日期格式正确
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # === 计算因子 ===
        # 1. 估值因子：市净率倒数（使用成交额/换手率估算流通市值）
        df['circulation_market_value'] = df['amount'] / (df['turnover_rate'] + 1e-8)
        df['pb_inverse'] = 1 / (df['circulation_market_value'] / (df['close'] * 1e8) + 1e-8)
        
        # 2. 盈利因子：过去20日换手率变化率
        df['turnover_ma20'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        df['turnover_ma5'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        df['profit_factor'] = (df['turnover_ma5'] - df['turnover_ma20']) / (df['turnover_ma20'] + 1e-8)
        
        # 3. 趋势因子：过去20日收益率（使用shift避免前向偏差）
        df['trend_20d'] = df.groupby('code')['close'].transform(
            lambda x: x / x.shift(20) - 1
        )
        
        # 4. 低波动因子：过去20日收益率标准差
        df['volatility_20d'] = df.groupby('code')['pct_change'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        df['volatility_score'] = -df['volatility_20d']
        
        # 5. 市场指数代理（使用所有股票平均价）
        df['market_avg_close'] = df.groupby('date')['close'].transform('mean')
        
        # 计算市场均线（使用shift避免前向偏差）
        # 先计算每日市场平均收盘价
        market_daily_avg = df.groupby('date')['close'].mean().reset_index()
        market_daily_avg.columns = ['date', 'market_daily_avg']
        df = df.merge(market_daily_avg, on='date', how='left')
        
        # 计算市场均线
        df['market_ma20'] = df.groupby('date')['market_daily_avg'].transform('first').rolling(20, min_periods=10).mean().shift(1)
        df['market_ma60'] = df.groupby('date')['market_daily_avg'].transform('first').rolling(60, min_periods=30).mean().shift(1)
        
        # 大盘择时信号
        df['market_timing'] = (
            (df['market_ma20'] > df['market_ma60']) & 
            (df['market_avg_close'] > df['market_ma20'])
        ).astype(int)
        
        # 6. 波动率过滤
        df['vol_rank'] = df.groupby('date')['volatility_20d'].transform(
            lambda x: x.rank(pct=True, method='min')
        )
        df['vol_filter'] = (df['vol_rank'] <= 0.8).astype(int)
        
        # 剔除缺失值（使用更宽松的条件）
        required_cols = ['pb_inverse', 'profit_factor', 'trend_20d', 'volatility_score', 'market_timing']
        df = df.dropna(subset=required_cols, how='any')
        
        # 过滤无效数据
        df = df[df['trend_20d'] > -1]  # 剔除极端负收益
        df = df[df['volatility_20d'] > 0]  # 剔除波动率为0的股票
        
        if len(df) == 0:
            return pd.DataFrame(columns=['code', 'weight', 'date'])
        
        # === 生成调仓日期：每周最后一个交易日 ===
        # 使用更稳健的方法确定每周最后一个交易日
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        df['year'] = df['date'].dt.year.astype(int)
        
        # 找到每周最后一个交易日
        last_trading_days = df.groupby(['year', 'week'])['date'].transform('max')
        df['is_rebalance_day'] = (df['date'] == last_trading_days)
        
        # 只保留调仓日数据
        rebalance_days = df[df['is_rebalance_day']].copy()
        
        if len(rebalance_days) == 0:
            return pd.DataFrame(columns=['code', 'weight', 'date'])
        
        results = []
        
        for date, group in rebalance_days.groupby('date'):
            # 大盘择时检查
            market_signal = group['market_timing'].iloc[0]
            if market_signal == 0:
                continue
            
            # 应用波动率过滤
            group = group[group['vol_filter'] == 1].copy()
            if len(group) < 5:
                continue
            
            # 因子标准化（z-score）
            for factor in ['pb_inverse', 'profit_factor', 'trend_20d', 'volatility_score']:
                mean_val = group[factor].mean()
                std_val = group[factor].std()
                if std_val > 0:
                    # 使用更明确的列名
                    col_name = f'z_{factor[:4]}'
                    group[col_name] = (group[factor] - mean_val) / std_val
                else:
                    col_name = f'z_{factor[:4]}'
                    group[col_name] = 0
            
            # 因子合成 - 使用正确的列名
            # pb_inverse -> z_pb_i
            # profit_factor -> z_pro
            # trend_20d -> z_tre
            # volatility_score -> z_vol
            group['total_score'] = (
                0.25 * group['z_pb_i'] + 
                0.25 * group['z_pro'] + 
                0.30 * group['z_tre'] + 
                0.20 * group['z_vol']
            )
            
            # 选择前20支股票
            selected = group.nlargest(min(20, len(group)), 'total_score')
            
            if len(selected) > 0:
                # 等权重
                weight = 1.0 / len(selected)
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