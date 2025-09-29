from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from loguru import logger

# 导入路由
from routers import chat, prompts, config, logs
from services.storage_service import StorageService
from services.langchain_service import LangChainService

# 加载环境变量
load_dotenv()

# 全局服务实例
storage_service = None
langchain_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global storage_service, langchain_service

    # 启动时初始化服务
    logger.info("正在初始化服务...")

    # 初始化存储服务
    storage_service = StorageService()
    await storage_service.initialize()

    # 初始化 LangChain 服务
    langchain_service = LangChainService(storage_service)
    await langchain_service.initialize()

    # 将服务实例添加到应用状态
    app.state.storage_service = storage_service
    app.state.langchain_service = langchain_service

    logger.info("服务初始化完成")

    yield

    # 关闭时清理资源
    logger.info("正在关闭服务...")
    if storage_service:
        await storage_service.close()
    logger.info("服务关闭完成")


# 创建 FastAPI 应用
app = FastAPI(
    title="LangChain 知识库助手 API",
    description="基于 LangChain 的智能知识库助手后端服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/chat", tags=["聊天"])
app.include_router(prompts.router, prefix="/api/prompts", tags=["Prompt模板"])
app.include_router(config.router, prefix="/api/config", tags=["系统配置"])
app.include_router(logs.router, prefix="/api/logs", tags=["日志管理"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LangChain 知识库助手 API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "services": {
            "storage": storage_service is not None,
            "langchain": langchain_service is not None,
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(status_code=500, content={"detail": "内部服务器错误"})


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8080))

    logger.info(f"启动服务器: {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")
