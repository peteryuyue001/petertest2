"""回测路由 — REST + WebSocket 实时推送"""

import asyncio
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from web.services.bridge import run_backtest_for_version

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    versions: List[int]  # 要回测的策略版本列表


@router.post("")
async def trigger_backtest(req: BacktestRequest):
    """
    触发回测（同步版本，适合快速单策略回测）
    返回简要结果摘要
    """
    results = []
    for version in req.versions:
        r = run_backtest_for_version(version)
        if r.get("success"):
            results.append({
                "version": version,
                "sharpe_ratio": r["metrics"].get("sharpe_ratio", 0),
                "total_return": r["metrics"].get("total_return", 0),
                "annual_return": r["metrics"].get("annual_return", 0),
                "max_drawdown": r["metrics"].get("max_drawdown", 0),
            })
        else:
            results.append({"version": version, "error": r.get("error", "未知错误")})
    return {"results": results}


# WebSocket 连接管理
_ws_clients: Dict[str, WebSocket] = {}


@router.websocket("/ws/{version}")
async def backtest_ws(websocket: WebSocket, version: int):
    """
    WebSocket 实时回测进度推送
    
    连接后自动开始回测，实时推送进度消息：
    - {"type": "progress", "message": "..."}
    - {"type": "result", "data": {...}}
    - {"type": "error", "message": "..."}
    - {"type": "completed"}
    """
    await websocket.accept()
    
    client_id = f"backtest_v{version}_{id(websocket)}"
    _ws_clients[client_id] = websocket

    try:
        # 定义进度回调（同步转异步推送）
        async def push_progress(msg):
            try:
                await websocket.send_json({"type": "progress", "message": msg})
            except Exception:
                pass

        def sync_progress(msg: str):
            # 在同步函数中安排异步任务
            asyncio.create_task(push_progress(msg))

        # 在线程池中执行回测（避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: run_backtest_for_version(version, progress_callback=sync_progress)
        )

        if result.get("success"):
            await websocket.send_json({
                "type": "result",
                "version": version,
                "metrics": result["metrics"],
                "equity_curve": result.get("equity_curve", []),
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": result.get("error", "回测失败"),
                "phase": result.get("phase", "unknown"),
            })

        await websocket.send_json({"type": "completed"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        _ws_clients.pop(client_id, None)