"""策略构建器路由 — 前端因子选择和策略代码生成"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

from web.services.strategy_builder import (
    get_factor_list,
    get_frequency_options,
    generate_strategy_code,
    save_generated_strategy,
)

router = APIRouter(prefix="/api/builder", tags=["builder"])


# ── 请求/响应模型 ──

class FactorConfig(BaseModel):
    """单个因子配置"""
    id: str
    weight: float = 1.0
    params: Dict[str, Any] = {}


class BuildRequest(BaseModel):
    """策略构建请求"""
    strategy_name: str = "自定义策略"
    description: str = ""
    factors: List[FactorConfig] = []
    holding_count: int = 15
    rebalance_freq: str = "monthly"
    enable_timing: bool = False
    timing_ma: int = 20
    filter_listed_days: int = 60


class PreviewRequest(BaseModel):
    """预览代码请求（与 BuildRequest 相同）"""
    strategy_name: str = "自定义策略"
    description: str = ""
    factors: List[FactorConfig] = []
    holding_count: int = 15
    rebalance_freq: str = "monthly"
    enable_timing: bool = False
    timing_ma: int = 20
    filter_listed_days: int = 60


# ── API 端点 ──

@router.get("/factors")
async def list_factors():
    """获取所有可用因子定义"""
    return {
        "factors": get_factor_list(),
        "frequencies": get_frequency_options(),
    }


@router.post("/preview")
async def preview_code(req: PreviewRequest):
    """预览生成的策略代码（不保存）"""
    if not req.factors:
        raise HTTPException(status_code=400, detail="请至少选择一个因子")

    config = req.model_dump()
    try:
        code = generate_strategy_code(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"代码生成失败: {str(e)}")

    return {"code": code}


@router.post("/save")
async def save_strategy(req: BuildRequest):
    """生成并保存策略到策略池"""
    if not req.factors:
        raise HTTPException(status_code=400, detail="请至少选择一个因子")

    config = req.model_dump()
    try:
        result = save_generated_strategy(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"策略保存失败: {str(e)}")

    return result