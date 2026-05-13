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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from web.routers import strategies, backtest, results, evolution
from web.services.bridge import get_system_status

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


# ── 根路径 ──

@app.get("/api/status")
async def status():
    """系统状态"""
    return get_system_status()


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