"""AI 进化路由 — 策略生成、改进、修复"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from web.services.bridge import (
    run_generate_strategy,
    run_improve_strategy,
    run_fix_strategy,
)

router = APIRouter(prefix="/api/evolve", tags=["evolution"])


class GenerateRequest(BaseModel):
    instruction: str = "生成一个沪深300多因子选股策略"


class FixRequest(BaseModel):
    error_message: str


@router.post("/generate")
async def generate_strategy(req: GenerateRequest):
    """AI 生成新策略"""
    data, error = run_generate_strategy(req.instruction)
    if error:
        raise HTTPException(status_code=500, detail=error)
    return data


@router.post("/improve/{version}")
async def improve_strategy(version: int):
    """AI 根据回测结果改进策略"""
    data, error = run_improve_strategy(version)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return data


@router.post("/fix/{version}")
async def fix_strategy(version: int, req: FixRequest):
    """AI 根据错误信息修复策略"""
    data, error = run_fix_strategy(version, req.error_message)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return data