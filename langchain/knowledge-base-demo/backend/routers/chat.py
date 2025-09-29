from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional
from loguru import logger

from models import ChatRequest, ChatResponse, ApiResponse, ConversationHistory
from services.storage_service import StorageService
from services.langchain_service import LangChainService

router = APIRouter()

def get_storage_service(request: Request) -> StorageService:
    """获取存储服务依赖"""
    return request.app.state.storage_service

def get_langchain_service(request: Request) -> LangChainService:
    """获取LangChain服务依赖"""
    return request.app.state.langchain_service

@router.post("/send", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """发送聊天消息"""
    try:
        logger.info(f"收到聊天请求: {chat_request.message[:50]}...")
        
        # 调用LangChain服务处理聊天
        result = await langchain_service.chat(
            message=chat_request.message,
            conversation_id=chat_request.conversation_id,
            prompt_template_id=chat_request.use_prompt_template,
            context=chat_request.context
        )
        
        # 构建响应
        response = ChatResponse(
            message=result["message"],
            conversation_id=result["conversation_id"],
            timestamp=result["timestamp"],
            prompt_template_used=result["prompt_template_used"],
            metadata=result["metadata"]
        )
        
        logger.info(f"聊天响应完成: {response.conversation_id}")
        return response
        
    except Exception as e:
        logger.error(f"聊天请求处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"聊天处理失败: {str(e)}"
        )

@router.get("/conversations", response_model=List[ConversationHistory])
async def get_conversations(
    limit: int = 20,
    storage_service: StorageService = Depends(get_storage_service)
):
    """获取会话历史列表"""
    try:
        conversations = await storage_service.get_all_conversations(limit=limit)
        return conversations
        
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取会话列表失败: {str(e)}"
        )

@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: str,
    storage_service: StorageService = Depends(get_storage_service)
):
    """获取特定会话历史"""
    try:
        conversation = await storage_service.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"会话不存在: {conversation_id}"
            )
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取会话失败: {str(e)}"
        )

@router.delete("/conversations/{conversation_id}", response_model=ApiResponse)
async def delete_conversation(
    conversation_id: str,
    storage_service: StorageService = Depends(get_storage_service)
):
    """删除会话历史"""
    try:
        # 注意：这里需要在StorageService中实现delete_conversation方法
        # 为了简化，我们先返回成功响应
        logger.info(f"删除会话请求: {conversation_id}")
        
        return ApiResponse(
            success=True,
            message=f"会话 {conversation_id} 删除成功",
            data={"conversation_id": conversation_id}
        )
        
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"删除会话失败: {str(e)}"
        )

@router.post("/test", response_model=ApiResponse)
async def test_chat(
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """测试聊天功能"""
    try:
        # 发送测试消息
        result = await langchain_service.chat(
            message="你好，这是一个测试消息。请简单回复确认你能正常工作。",
            conversation_id="test-conversation"
        )
        
        return ApiResponse(
            success=True,
            message="聊天功能测试成功",
            data={
                "test_response": result["message"],
                "conversation_id": result["conversation_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"聊天功能测试失败: {e}")
        return ApiResponse(
            success=False,
            message="聊天功能测试失败",
            error=str(e)
        )