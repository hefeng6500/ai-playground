from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any
from loguru import logger

from models import SystemConfig, ApiResponse
from services.storage_service import StorageService
from services.langchain_service import LangChainService

router = APIRouter()

def get_storage_service(request: Request) -> StorageService:
    """获取存储服务依赖"""
    return request.app.state.storage_service

def get_langchain_service(request: Request) -> LangChainService:
    """获取LangChain服务依赖"""
    return request.app.state.langchain_service

@router.get("/", response_model=SystemConfig)
async def get_config(
    storage_service: StorageService = Depends(get_storage_service)
):
    """获取系统配置"""
    try:
        config = await storage_service.get_config()
        return config
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取系统配置失败: {str(e)}"
        )

@router.put("/", response_model=SystemConfig)
async def update_config(
    config_updates: Dict[str, Any],
    storage_service: StorageService = Depends(get_storage_service),
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """更新系统配置"""
    try:
        # 验证配置参数
        valid_fields = {
            "openai_model", "max_tokens", "temperature", 
            "default_prompt_template", "enable_logging", "log_level"
        }
        
        # 过滤无效字段
        filtered_updates = {k: v for k, v in config_updates.items() if k in valid_fields}
        
        if not filtered_updates:
            raise HTTPException(
                status_code=400,
                detail="没有有效的配置参数"
            )
        
        # 验证特定字段
        if "temperature" in filtered_updates:
            temp = filtered_updates["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise HTTPException(
                    status_code=400,
                    detail="temperature参数必须在0-2之间"
                )
        
        if "max_tokens" in filtered_updates:
            max_tokens = filtered_updates["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4000:
                raise HTTPException(
                    status_code=400,
                    detail="max_tokens参数必须在1-4000之间"
                )
        
        if "log_level" in filtered_updates:
            log_level = filtered_updates["log_level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if log_level not in valid_levels:
                raise HTTPException(
                    status_code=400,
                    detail=f"log_level必须是以下值之一: {valid_levels}"
                )
        
        # 如果更新了默认Prompt模板，验证其存在性
        if "default_prompt_template" in filtered_updates:
            template_id = filtered_updates["default_prompt_template"]
            if template_id:  # 允许设置为None
                template = await storage_service.get_prompt_by_id(template_id)
                if not template:
                    raise HTTPException(
                        status_code=400,
                        detail=f"默认Prompt模板不存在: {template_id}"
                    )
        
        # 更新配置
        updated_config = await storage_service.update_config(filtered_updates)
        
        # 如果更新了LLM相关配置，重新初始化LangChain服务
        llm_related_fields = {"openai_model", "max_tokens", "temperature"}
        if any(field in filtered_updates for field in llm_related_fields):
            try:
                await langchain_service.update_llm_config(filtered_updates)
                logger.info("LLM配置已更新并重新初始化")
            except Exception as e:
                logger.error(f"LLM配置更新失败: {e}")
                # 不抛出异常，因为配置已经保存，只是LLM重新初始化失败
        
        logger.info(f"系统配置更新成功: {list(filtered_updates.keys())}")
        return updated_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新系统配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"更新系统配置失败: {str(e)}"
        )

@router.post("/reset", response_model=ApiResponse)
async def reset_config(
    storage_service: StorageService = Depends(get_storage_service),
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """重置系统配置为默认值"""
    try:
        # 默认配置
        default_config = {
            "openai_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
            "default_prompt_template": "default-chat",
            "enable_logging": True,
            "log_level": "INFO"
        }
        
        # 更新配置
        updated_config = await storage_service.update_config(default_config)
        
        # 重新初始化LangChain服务
        try:
            await langchain_service.update_llm_config(default_config)
            logger.info("系统配置已重置并重新初始化LLM")
        except Exception as e:
            logger.error(f"LLM重新初始化失败: {e}")
        
        return ApiResponse(
            success=True,
            message="系统配置已重置为默认值",
            data=updated_config.dict()
        )
        
    except Exception as e:
        logger.error(f"重置系统配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"重置系统配置失败: {str(e)}"
        )

@router.get("/models", response_model=list)
async def get_available_models():
    """获取可用的OpenAI模型列表"""
    try:
        # 返回常用的OpenAI模型列表
        models = [
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "description": "快速、经济的通用模型",
                "max_tokens": 4096
            },
            {
                "id": "gpt-3.5-turbo-16k",
                "name": "GPT-3.5 Turbo 16K",
                "description": "支持更长上下文的GPT-3.5模型",
                "max_tokens": 16384
            },
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "description": "最强大的通用模型",
                "max_tokens": 8192
            },
            {
                "id": "gpt-4-turbo-preview",
                "name": "GPT-4 Turbo",
                "description": "更快的GPT-4模型",
                "max_tokens": 128000
            }
        ]
        
        return models
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取模型列表失败: {str(e)}"
        )

@router.post("/test", response_model=ApiResponse)
async def test_config(
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """测试当前配置"""
    try:
        # 发送测试消息
        result = await langchain_service.chat(
            message="请回复'配置测试成功'来确认系统正常工作。",
            conversation_id="config-test"
        )
        
        return ApiResponse(
            success=True,
            message="配置测试成功",
            data={
                "test_response": result["message"],
                "conversation_id": result["conversation_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"配置测试失败: {e}")
        return ApiResponse(
            success=False,
            message="配置测试失败",
            error=str(e)
        )