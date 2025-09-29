from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息")
    conversation_id: Optional[str] = Field(None, description="会话ID")
    use_prompt_template: Optional[str] = Field(None, description="使用的Prompt模板ID")
    context: Optional[Dict[str, Any]] = Field(None, description="额外上下文")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    message: str = Field(..., description="助手回复")
    conversation_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(..., description="响应时间")
    prompt_template_used: Optional[str] = Field(None, description="使用的Prompt模板")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class PromptTemplate(BaseModel):
    """Prompt模板模型"""
    id: str = Field(..., description="模板ID")
    name: str = Field(..., description="模板名称")
    description: Optional[str] = Field(None, description="模板描述")
    template: str = Field(..., description="模板内容")
    variables: List[str] = Field(default_factory=list, description="模板变量")
    category: Optional[str] = Field(None, description="模板分类")
    is_active: bool = Field(True, description="是否激活")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

class PromptTemplateCreate(BaseModel):
    """创建Prompt模板请求"""
    name: str = Field(..., description="模板名称")
    description: Optional[str] = Field(None, description="模板描述")
    template: str = Field(..., description="模板内容")
    variables: List[str] = Field(default_factory=list, description="模板变量")
    category: Optional[str] = Field(None, description="模板分类")

class PromptTemplateUpdate(BaseModel):
    """更新Prompt模板请求"""
    name: Optional[str] = Field(None, description="模板名称")
    description: Optional[str] = Field(None, description="模板描述")
    template: Optional[str] = Field(None, description="模板内容")
    variables: Optional[List[str]] = Field(None, description="模板变量")
    category: Optional[str] = Field(None, description="模板分类")
    is_active: Optional[bool] = Field(None, description="是否激活")

class SystemConfig(BaseModel):
    """系统配置模型"""
    openai_model: str = Field("gpt-3.5-turbo", description="OpenAI模型")
    max_tokens: int = Field(1000, description="最大token数")
    temperature: float = Field(0.7, description="温度参数")
    default_prompt_template: Optional[str] = Field(None, description="默认Prompt模板ID")
    enable_logging: bool = Field(True, description="是否启用日志")
    log_level: str = Field("INFO", description="日志级别")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

class LogEntry(BaseModel):
    """日志条目模型"""
    id: str = Field(..., description="日志ID")
    level: str = Field(..., description="日志级别")
    message: str = Field(..., description="日志消息")
    timestamp: datetime = Field(..., description="时间戳")
    source: Optional[str] = Field(None, description="日志来源")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class ConversationHistory(BaseModel):
    """会话历史模型"""
    conversation_id: str = Field(..., description="会话ID")
    messages: List[ChatMessage] = Field(..., description="消息列表")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class ApiResponse(BaseModel):
    """通用API响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")