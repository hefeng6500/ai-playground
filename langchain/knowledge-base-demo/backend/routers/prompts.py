from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional
from loguru import logger

from models import PromptTemplate, PromptTemplateCreate, PromptTemplateUpdate, ApiResponse
from services.storage_service import StorageService
from services.langchain_service import LangChainService

router = APIRouter()

def get_storage_service(request: Request) -> StorageService:
    """获取存储服务依赖"""
    return request.app.state.storage_service

def get_langchain_service(request: Request) -> LangChainService:
    """获取LangChain服务依赖"""
    return request.app.state.langchain_service

@router.get("/", response_model=List[PromptTemplate])
async def get_all_prompts(
    category: Optional[str] = None,
    active_only: bool = True,
    storage_service: StorageService = Depends(get_storage_service)
):
    """获取所有Prompt模板"""
    try:
        prompts = await storage_service.get_all_prompts()
        
        # 按分类过滤
        if category:
            prompts = [p for p in prompts if p.category == category]
        
        # 只返回激活的模板
        if active_only:
            prompts = [p for p in prompts if p.is_active]
        
        logger.info(f"返回 {len(prompts)} 个Prompt模板")
        return prompts
        
    except Exception as e:
        logger.error(f"获取Prompt模板失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取Prompt模板失败: {str(e)}"
        )

@router.get("/{prompt_id}", response_model=PromptTemplate)
async def get_prompt(
    prompt_id: str,
    storage_service: StorageService = Depends(get_storage_service)
):
    """获取特定Prompt模板"""
    try:
        prompt = await storage_service.get_prompt_by_id(prompt_id)
        
        if not prompt:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt模板不存在: {prompt_id}"
            )
        
        return prompt
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取Prompt模板失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取Prompt模板失败: {str(e)}"
        )

@router.post("/", response_model=PromptTemplate)
async def create_prompt(
    prompt_data: PromptTemplateCreate,
    storage_service: StorageService = Depends(get_storage_service),
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """创建新的Prompt模板"""
    try:
        # 验证模板格式
        validation_result = await langchain_service.validate_prompt_template(
            prompt_data.template,
            prompt_data.variables
        )
        
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt模板格式无效: {validation_result['message']}"
            )
        
        # 创建模板
        prompt = await storage_service.create_prompt(prompt_data.dict())
        
        logger.info(f"创建Prompt模板成功: {prompt.name} (ID: {prompt.id})")
        return prompt
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建Prompt模板失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"创建Prompt模板失败: {str(e)}"
        )

@router.put("/{prompt_id}", response_model=PromptTemplate)
async def update_prompt(
    prompt_id: str,
    prompt_data: PromptTemplateUpdate,
    storage_service: StorageService = Depends(get_storage_service),
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """更新Prompt模板"""
    try:
        # 检查模板是否存在
        existing_prompt = await storage_service.get_prompt_by_id(prompt_id)
        if not existing_prompt:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt模板不存在: {prompt_id}"
            )
        
        # 如果更新了模板内容，需要验证
        if prompt_data.template is not None:
            variables = prompt_data.variables if prompt_data.variables is not None else existing_prompt.variables
            validation_result = await langchain_service.validate_prompt_template(
                prompt_data.template,
                variables
            )
            
            if not validation_result["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt模板格式无效: {validation_result['message']}"
                )
        
        # 更新模板
        update_dict = {k: v for k, v in prompt_data.dict().items() if v is not None}
        updated_prompt = await storage_service.update_prompt(prompt_id, update_dict)
        
        if not updated_prompt:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt模板不存在: {prompt_id}"
            )
        
        logger.info(f"更新Prompt模板成功: {updated_prompt.name} (ID: {prompt_id})")
        return updated_prompt
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新Prompt模板失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"更新Prompt模板失败: {str(e)}"
        )

@router.delete("/{prompt_id}", response_model=ApiResponse)
async def delete_prompt(
    prompt_id: str,
    storage_service: StorageService = Depends(get_storage_service)
):
    """删除Prompt模板"""
    try:
        # 检查模板是否存在
        existing_prompt = await storage_service.get_prompt_by_id(prompt_id)
        if not existing_prompt:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt模板不存在: {prompt_id}"
            )
        
        # 删除模板
        success = await storage_service.delete_prompt(prompt_id)
        
        if success:
            logger.info(f"删除Prompt模板成功: {existing_prompt.name} (ID: {prompt_id})")
            return ApiResponse(
                success=True,
                message=f"Prompt模板 '{existing_prompt.name}' 删除成功",
                data={"prompt_id": prompt_id}
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="删除Prompt模板失败"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除Prompt模板失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"删除Prompt模板失败: {str(e)}"
        )

@router.post("/{prompt_id}/validate", response_model=ApiResponse)
async def validate_prompt(
    prompt_id: str,
    test_variables: dict = None,
    storage_service: StorageService = Depends(get_storage_service),
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """验证Prompt模板"""
    try:
        # 获取模板
        prompt = await storage_service.get_prompt_by_id(prompt_id)
        if not prompt:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt模板不存在: {prompt_id}"
            )
        
        # 验证模板
        validation_result = await langchain_service.validate_prompt_template(
            prompt.template,
            prompt.variables
        )
        
        return ApiResponse(
            success=validation_result["valid"],
            message=validation_result["message"],
            data={
                "prompt_id": prompt_id,
                "validation_result": validation_result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证Prompt模板失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"验证Prompt模板失败: {str(e)}"
        )

@router.get("/categories/list", response_model=List[str])
async def get_prompt_categories(
    storage_service: StorageService = Depends(get_storage_service)
):
    """获取所有Prompt模板分类"""
    try:
        prompts = await storage_service.get_all_prompts()
        categories = list(set(p.category for p in prompts if p.category))
        categories.sort()
        
        return categories
        
    except Exception as e:
        logger.error(f"获取Prompt分类失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取Prompt分类失败: {str(e)}"
        )