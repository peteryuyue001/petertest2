#!/usr/bin/env python3
"""
Quant Evolution Web — FastAPI 应用入口

启动: python -m web.app
      或: cd quant_evolution && uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from web.routers import strategies, backtest, results, evolution, builder
from web.services.bridge import get_system_status, get_data_status, run_fetch_data_web

# ── 应用初始化 ──

app = FastAPI(
    title="Quant Evolution API",
    description="量化 AI 进化系统 Web 接口",
    version="1.0.0",
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ──

app.include_router(strategies.router)
app.include_router(backtest.router)
app.include_router(results.router)
app.include_router(evolution.router)
app.include_router(builder.router)


# ── 根路径 ──

@app.get("/api/status")
async def status():
    """系统状态"""
    return get_system_status()


@app.get("/api/data/status")
async def data_status():
    """数据缓存状态"""
    return get_data_status()


class FetchRequest(BaseModel):
    stock_pool: str = "hs300"


@app.post("/api/data/fetch")
async def fetch_data(req: FetchRequest):
    """触发数据下载（同步，小规模数据快速返回）"""
    from threading import Lock
    result = run_fetch_data_web(req.stock_pool)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "数据下载失败"))
    return result


@app.websocket("/ws/data/fetch/{stock_pool}")
async def fetch_data_ws(websocket: WebSocket, stock_pool: str):
    """WebSocket 实时数据下载进度推送"""
    await websocket.accept()
    try:
        from web.services.bridge import run_fetch_data_web
        loop = asyncio.get_event_loop()
        
        async def push_progress(msg):
            try:
                await websocket.send_json({"type": "progress", "message": msg})
            except Exception:
                pass
        
        def sync_progress(msg: str):
            asyncio.run_coroutine_threadsafe(push_progress(msg), loop)
        
        result = await loop.run_in_executor(
            None, lambda: run_fetch_data_web(stock_pool, progress_callback=sync_progress)
        )
        if result.get("success"):
            await websocket.send_json({"type": "completed", "data": result})
        else:
            await websocket.send_json({"type": "error", "message": result.get("error", "未知错误")})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── 静态文件（前端 HTML/JS/CSS） ──

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


# ── 直接运行 ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )