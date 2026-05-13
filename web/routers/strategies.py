"""策略管理路由"""

from fastapi import APIRouter, HTTPException
from web.services.bridge import get_strategies, get_strategy_code, delete_strategy

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("")
async def list_strategies():
    """获取所有策略列表"""
    return get_strategies()


@router.get("/{version}")
async def view_strategy(version: int):
    """获取策略源代码"""
    code, error = get_strategy_code(version)
    if error:
        raise HTTPException(status_code=404, detail=error)
    return {"version": version, "code": code}


@router.delete("/{version}")
async def remove_strategy(version: int):
    """删除策略及其回测结果"""
    success, message = delete_strategy(version)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    return {"message": message}