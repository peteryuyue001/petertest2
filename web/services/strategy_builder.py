"""
策略构建器 — 因子定义 + 策略代码动态生成

用户在前端选择因子、配置参数后，此模块将配置转换为可执行的策略 Python 代码。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 因子定义
# ============================================================

@dataclass
class FactorParam:
    """因子参数定义"""
    key: str                # 参数键名
    label: str              # 中文标签
    type: str               # 类型: 'int', 'float', 'select'
    default: Any = None     # 默认值
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict]] = None  # select 类型的选项


@dataclass
class FactorDefinition:
    """因子定义"""
    id: str                          # 因子唯一 ID
    name: str                        # 因子中文名称
    category: str                    # 分类: 'momentum', 'scale', 'volatility', 'liquidity', 'value'
    description: str                 # 因子描述
    params: List[FactorParam] = field(default_factory=list)
    # 代码生成模板（{factor_var} 会被替换为实际变量名）
    calc_template: str = ""
    zscore_template: str = "{factor_var}_z = df.groupby('date')['{factor_var}'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))"


# 所有可用因子定义
FACTOR_DEFINITIONS: List[FactorDefinition] = [
    FactorDefinition(
        id="reversal",
        name="反转因子",
        category="momentum",
        description="过去N日涨跌幅（取负值，跌幅越大得分越高）",
        params=[
            FactorParam(key="window", label="回看窗口(日)", type="int", default=20, min=5, max=120, step=5),
        ],
        calc_template="""
        # 反转因子：过去{window}日涨跌幅（取负值，超跌预期反转）
        df['{factor_var}'] = -df.groupby('code')['close'].transform(
            lambda x: x.pct_change({window}).shift(1))
        """,
    ),
    FactorDefinition(
        id="ma_deviation",
        name="均线偏离因子",
        category="momentum",
        description="当前价格相对N日均线的偏离程度（超跌得分高）",
        params=[
            FactorParam(key="ma_period", label="均线周期(日)", type="int", default=60, min=10, max=250, step=10),
        ],
        calc_template="""
        # 均线偏离因子：价格偏离{ma_period}日均线（取负值，超跌得分高）
        df['ma_{ma_period}'] = df.groupby('code')['close'].transform(
            lambda x: x.rolling({ma_period}).mean().shift(1))
        df['{factor_var}'] = -((df['close'] / df['ma_{ma_period}']) - 1)
        """,
    ),
    FactorDefinition(
        id="small_cap",
        name="小市值因子",
        category="scale",
        description="市值越小得分越高（通过成交额/换手率估算）",
        params=[],
        calc_template="""
        # 小市值因子：通过成交额/换手率估算市值，取负值
        df['{factor_var}_raw'] = df['amount'] / (df['turnover_rate'] + 0.001)
        df['{factor_var}'] = -df['{factor_var}_raw']
        """,
    ),
    FactorDefinition(
        id="low_volatility",
        name="低波动因子",
        category="volatility",
        description="过去N日收益率波动率（越低得分越高）",
        params=[
            FactorParam(key="window", label="回看窗口(日)", type="int", default=20, min=5, max=120, step=5),
        ],
        calc_template="""
        # 低波动因子：过去{window}日收益率标准差（取负值，低波动得分高）
        df['{factor_var}_raw'] = df.groupby('code')['pct_change'].transform(
            lambda x: x.rolling({window}).std().shift(1))
        df['{factor_var}'] = -df['{factor_var}_raw']
        """,
    ),
    FactorDefinition(
        id="low_turnover",
        name="低换手因子",
        category="liquidity",
        description="过去N日平均换手率（越低得分越高）",
        params=[
            FactorParam(key="window", label="回看窗口(日)", type="int", default=20, min=5, max=120, step=5),
        ],
        calc_template="""
        # 低换手因子：过去{window}日平均换手率（取负值，低换手得分高）
        df['{factor_var}'] = -df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling({window}).mean().shift(1))
        """,
    ),
    FactorDefinition(
        id="low_value",
        name="低估值因子",
        category="value",
        description="价格越低估值越低得分越高（简化版）",
        params=[],
        calc_template="""
        # 低估值因子：价格倒数（价格越低得分越高）
        df['{factor_var}'] = 1.0 / (df['close'] + 0.01)
        """,
    ),
    FactorDefinition(
        id="volume_ratio",
        name="量比因子",
        category="momentum",
        description="过去5日均量相对20日均量的比值（放量上涨信号）",
        params=[],
        calc_template="""
        # 量比因子：5日均量 / 20日均量
        df['vol_ma5'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean().shift(1))
        df['vol_ma20'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean().shift(1))
        df['{factor_var}'] = df['vol_ma5'] / (df['vol_ma20'] + 0.01) - 1
        """,
    ),
    FactorDefinition(
        id="price_position",
        name="价格位置因子",
        category="momentum",
        description="当前价格在N日内的相对位置（0-1，低位得分高）",
        params=[
            FactorParam(key="window", label="回看窗口(日)", type="int", default=60, min=10, max=250, step=10),
        ],
        calc_template="""
        # 价格位置因子：当前价在{window}日内的相对位置（低位得分高）
        df['pp_high'] = df.groupby('code')['high'].transform(
            lambda x: x.rolling({window}).max().shift(1))
        df['pp_low'] = df.groupby('code')['low'].transform(
            lambda x: x.rolling({window}).min().shift(1))
        df['{factor_var}'] = -((df['close'] - df['pp_low']) / (df['pp_high'] - df['pp_low'] + 0.01))
        """,
    ),
    FactorDefinition(
        id="relative_strength",
        name="相对强度因子",
        category="momentum",
        description="过去N日涨幅减基准涨幅（相对强弱）",
        params=[
            FactorParam(key="window", label="回看窗口(日)", type="int", default=20, min=5, max=120, step=5),
        ],
        calc_template="""
        # 相对强度因子：个股{window}日涨幅 - 全市场均值涨幅
        df['{factor_var}_ret'] = df.groupby('code')['close'].transform(
            lambda x: x.pct_change({window}).shift(1))
        df['{factor_var}_avg'] = df.groupby('date')['{factor_var}_ret'].transform('mean')
        df['{factor_var}'] = df['{factor_var}_ret'] - df['{factor_var}_avg']
        """,
    ),
    FactorDefinition(
        id="max_drawdown_factor",
        name="最大回撤因子",
        category="volatility",
        description="过去N日最大回撤（回撤越小得分越高）",
        params=[
            FactorParam(key="window", label="回看窗口(日)", type="int", default=60, min=20, max=250, step=10),
        ],
        calc_template="""
        # 最大回撤因子：过去{window}日最大回撤（取负值，回撤小得分高）
        def rolling_max_dd(series, w):
            roll_max = series.rolling(w).max()
            dd = series / roll_max - 1
            return dd.rolling(w).min()
        df['{factor_var}'] = -df.groupby('code')['close'].transform(
            lambda x: rolling_max_dd(x, {window}).shift(1))
        """,
    ),
]


# ============================================================
# 调仓频率定义
# ============================================================

REBALANCE_FREQUENCIES = [
    {"value": "daily", "label": "每日", "code_ref": "D"},
    {"value": "weekly", "label": "每周", "code_ref": "W"},
    {"value": "monthly", "label": "月度", "code_ref": "M"},
]


# ============================================================
# 策略代码生成器
# ============================================================

def get_factor_list() -> List[Dict]:
    """返回所有可用因子定义（供前端 API 使用）"""
    result = []
    for f in FACTOR_DEFINITIONS:
        result.append({
            "id": f.id,
            "name": f.name,
            "category": f.category,
            "description": f.description,
            "params": [
                {
                    "key": p.key,
                    "label": p.label,
                    "type": p.type,
                    "default": p.default,
                    "min": p.min,
                    "max": p.max,
                    "step": p.step,
                    "options": p.options,
                }
                for p in f.params
            ],
        })
    return result


def get_frequency_options() -> List[Dict]:
    """返回调仓频率选项"""
    return REBALANCE_FREQUENCIES


def generate_strategy_code(config: Dict) -> str:
    """
    根据配置生成策略 Python 代码。

    配置格式:
    {
        "strategy_name": "我的策略",
        "description": "策略描述",
        "factors": [
            {"id": "reversal", "weight": 0.5, "params": {"window": 20}},
            {"id": "low_volatility", "weight": 0.5, "params": {"window": 20}},
        ],
        "holding_count": 15,
        "rebalance_freq": "monthly",
        "enable_timing": false,
        "timing_ma": 20,
        "filter_st": true,
        "filter_listed_days": 60
    }
    """
    strategy_name = config.get("strategy_name", "自定义策略")
    description = config.get("description", "通过策略构建器生成的策略")
    factors = config.get("factors", [])
    holding_count = config.get("holding_count", 15)
    rebalance_freq = config.get("rebalance_freq", "monthly")
    enable_timing = config.get("enable_timing", False)
    timing_ma = config.get("timing_ma", 20)
    filter_listed_days = config.get("filter_listed_days", 60)

    # 归一化权重
    total_weight = sum(f.get("weight", 0) for f in factors)
    for fcfg in factors:
        if total_weight > 0:
            fcfg["weight"] = fcfg["weight"] / total_weight

    # 找到选中因子的定义
    factor_defs = {}
    for fdef in FACTOR_DEFINITIONS:
        factor_defs[fdef.id] = fdef

    # 计算版本号
    next_version = _get_next_version()

    # 生成因子描述
    factor_desc_parts = []
    for fcfg in factors:
        fdef = factor_defs.get(fcfg["id"])
        if fdef:
            weight_pct = round(fcfg["weight"] * 100, 1)
            factor_desc_parts.append(f"{fdef.name}({weight_pct}%)")
    factor_desc = " + ".join(factor_desc_parts)

    freq_label = {"daily": "每日", "weekly": "每周", "monthly": "月度"}.get(rebalance_freq, "月度")

    # ── 构建代码 ──
    lines = []
    lines.append('import pandas as pd')
    lines.append('import numpy as np')
    lines.append('from template.strategy_base import BaseStrategy')
    lines.append('')
    lines.append(f'class CustomStrategy(BaseStrategy):')
    lines.append(f'    name = "{strategy_name}"')
    lines.append(f'    version = {next_version}')
    lines.append(f'    description = "{factor_desc}，{freq_label}调仓，持仓{holding_count}支"')
    lines.append('')
    lines.append('    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:')
    lines.append('        """')
    lines.append(f'        因子: {factor_desc}')
    lines.append(f'        调仓: {freq_label}')
    lines.append(f'        持仓: {holding_count}支')
    if enable_timing:
        lines.append(f'        择时: {timing_ma}日均线过滤')
    lines.append('        """')
    lines.append('        df = data.copy()')
    lines.append("        df['date'] = pd.to_datetime(df['date'])")
    lines.append("        df = df.sort_values(['code', 'date']).reset_index(drop=True)")
    lines.append('')

    # 生成因子计算代码
    factor_vars = []
    for idx, fcfg in enumerate(factors):
        fid = fcfg["id"]
        fdef = factor_defs.get(fid)
        if not fdef:
            continue
        fvar = f"factor_{fid}"
        factor_vars.append(fvar)
        params = fcfg.get("params", {})

        # 合并默认参数
        merged_params = {}
        for pdef in fdef.params:
            merged_params[pdef.key] = params.get(pdef.key, pdef.default)

        lines.append(f'        # ===== 因子{idx+1}: {fdef.name} (权重: {fcfg["weight"]:.2f}) =====')

        # 生成计算代码
        calc_code = fdef.calc_template.format(factor_var=fvar, **merged_params)
        for line in calc_code.strip().split('\n'):
            stripped = line.strip()
            if stripped:
                lines.append(f'        {stripped}')

        # 生成标准化代码
        zscore_code = fdef.zscore_template.format(factor_var=fvar)
        for line in zscore_code.strip().split('\n'):
            stripped = line.strip()
            if stripped:
                lines.append(f'        {stripped}')

        lines.append('')

    # 合成得分
    lines.append('        # ===== 合成综合得分 =====')
    composite_parts = []
    for idx, fcfg in enumerate(factors):
        fvar = f"factor_{fcfg['id']}"
        w = fcfg["weight"]
        if composite_parts:
            composite_parts.append(f' + df[\'{fvar}_z\'] * {w:.4f}')
        else:
            composite_parts.append(f'df[\'{fvar}_z\'] * {w:.4f}')
    lines.append(f'        df[\'composite_score\'] = { "".join(composite_parts)}')
    lines.append('')

    # 过滤条件
    lines.append('        # ===== 过滤条件 =====')
    lines.append("        df = df[df['turnover_rate'] > 0]  # 排除停牌")
    lines.append("        df = df[df['close'] > 0]")
    if filter_listed_days and filter_listed_days > 0:
        lines.append(f'        # 排除上市不足{filter_listed_days}日的股票')
        lines.append(f'        df[\'min_date\'] = df.groupby(\'code\')[\'date\'].transform(\'min\')')
        lines.append(f'        df[\'days_listed\'] = (df[\'date\'] - df[\'min_date\']).dt.days')
        lines.append(f'        df = df[df[\'days_listed\'] >= {filter_listed_days}]')
    lines.append('')

    # 大盘择时
    if enable_timing:
        lines.append('        # ===== 大盘择时 =====')
        lines.append("        index_df = df.groupby('date').agg({'close': 'mean'}).reset_index()")
        lines.append("        index_df = index_df.sort_values('date')")
        lines.append(f'        index_df[\'ma{timing_ma}\'] = index_df[\'close\'].rolling({timing_ma}).mean()')
        lines.append(f"        index_df['market_trend'] = index_df['close'] > index_df['ma{timing_ma}']")
        lines.append("        df = df.merge(index_df[['date', 'market_trend']], on='date', how='left')")
        lines.append('')

    # 调仓频率
    lines.append('        # ===== 调仓选股 =====')
    if rebalance_freq == "monthly":
        lines.append("        df['year_month'] = df['date'].dt.to_period('M')")
        lines.append("        df['is_rebalance'] = df.groupby('year_month')['date'].transform(lambda x: x == x.max())")
    elif rebalance_freq == "weekly":
        lines.append("        df['week'] = df['date'].dt.isocalendar().week")
        lines.append("        df['year'] = df['date'].dt.year")
        lines.append("        df['is_rebalance'] = df.groupby(['year', 'week'])['date'].transform(lambda x: x == x.max())")
    else:  # daily
        lines.append("        df['is_rebalance'] = True")
    lines.append('')

    if enable_timing:
        lines.append("        df = df[(df['is_rebalance']) & (df['market_trend'] == True)]")
    else:
        lines.append("        df = df[df['is_rebalance']]")

    lines.append('')

    # 选股
    lines.append(f'        # ===== 选取得分最高的{holding_count}支股票 =====')
    lines.append("        df = df.dropna(subset=['composite_score'])")
    lines.append("        df['rank'] = df.groupby('date')['composite_score'].rank(ascending=False)")
    lines.append(f'        df = df[df[\'rank\'] <= {holding_count}]')
    lines.append('')
    lines.append('        # ===== 构建输出 =====')
    lines.append(f"        df['weight'] = 1.0 / {holding_count}")
    lines.append("        result = df[['code', 'weight', 'date']].copy()")
    lines.append("        result = result.reset_index(drop=True)")
    lines.append("        result['weight'] = result['weight'].round(4)")
    lines.append('')
    lines.append('        return result')
    lines.append('')

    return '\n'.join(lines)


def save_generated_strategy(config: Dict) -> Dict:
    """生成并保存策略，返回版本信息"""
    code = generate_strategy_code(config)
    next_version = _get_next_version()
    path = PROJECT_ROOT / "strategy_pool" / f"strategy_v{next_version}.py"
    path.write_text(code, encoding="utf-8")
    return {
        "version": next_version,
        "filename": f"strategy_v{next_version}.py",
        "code": code,
    }


def _get_next_version() -> int:
    """获取下一个可用版本号"""
    pool_dir = PROJECT_ROOT / "strategy_pool"
    existing = list(pool_dir.glob("strategy_v*.py"))
    if not existing:
        return 1
    versions = []
    for f in existing:
        try:
            versions.append(int(f.stem.replace("strategy_v", "")))
        except ValueError:
            continue
    return max(versions) + 1 if versions else 1