"""回测结果路由"""

from fastapi import APIRouter, HTTPException
from web.services.bridge import (
    get_backtest_result,
    get_all_results,
    get_evolution_comparison,
)

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("")
async def list_results():
    """获取所有回测结果摘要"""
    return get_all_results()


@router.get("/comparison")
async def evolution_comparison():
    """获取策略进化对比数据（所有策略核心指标）"""
    return get_evolution_comparison()


@router.get("/{version}")
async def view_result(version: int):
    """获取指定策略的回测详细结果"""
    result, error = get_backtest_result(version)
    if error:
        raise HTTPException(status_code=404, detail=error)
    return result