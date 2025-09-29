import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate as LangChainPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

from models import ChatMessage, MessageRole, PromptTemplate
from services.storage_service import StorageService

class LoggingCallbackHandler(BaseCallbackHandler):
    """LangChain日志回调处理器"""
    
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """LLM开始调用时的回调"""
        await self.storage_service.add_log(
            "INFO",
            f"LLM调用开始: {serialized.get('name', 'Unknown')}",
            "langchain_service",
            {"prompts_count": len(prompts)}
        )
    
    async def on_llm_end(self, response, **kwargs):
        """LLM调用结束时的回调"""
        await self.storage_service.add_log(
            "INFO",
            "LLM调用完成",
            "langchain_service",
            {"response_length": len(str(response))}
        )
    
    async def on_llm_error(self, error: Exception, **kwargs):
        """LLM调用出错时的回调"""
        await self.storage_service.add_log(
            "ERROR",
            f"LLM调用出错: {str(error)}",
            "langchain_service",
            {"error_type": type(error).__name__}
        )

class LangChainService:
    """LangChain集成服务"""
    
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        self.llm = None
        self.callback_handler = None
    
    async def initialize(self):
        """初始化LangChain服务"""
        logger.info("初始化LangChain服务...")
        
        # 检查环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY环境变量未设置")
        
        # 获取系统配置
        config = await self.storage_service.get_config()
        
        # 初始化回调处理器
        self.callback_handler = LoggingCallbackHandler(self.storage_service)
        
        # 初始化OpenAI LLM
        llm_kwargs = {
            "model": config.openai_model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "openai_api_key": api_key,
            "callbacks": [self.callback_handler]
        }
        
        if base_url:
            llm_kwargs["openai_api_base"] = base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        logger.info(f"LangChain服务初始化完成 - 模型: {config.openai_model}")
        
        # 记录初始化日志
        await self.storage_service.add_log(
            "INFO",
            "LangChain服务初始化完成",
            "langchain_service",
            {"model": config.openai_model, "temperature": config.temperature}
        )
    
    async def chat(self, 
                   message: str, 
                   conversation_id: str = None,
                   prompt_template_id: str = None,
                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理聊天请求"""
        try:
            # 生成会话ID
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # 获取会话历史
            conversation_history = await self.storage_service.get_conversation(conversation_id)
            messages = conversation_history.messages if conversation_history else []
            
            # 处理Prompt模板
            if prompt_template_id:
                prompt_template = await self.storage_service.get_prompt_by_id(prompt_template_id)
                if prompt_template:
                    # 使用模板处理消息
                    formatted_message = await self._format_message_with_template(
                        message, prompt_template, context
                    )
                else:
                    logger.warning(f"未找到Prompt模板: {prompt_template_id}")
                    formatted_message = message
            else:
                # 使用默认模板
                config = await self.storage_service.get_config()
                if config.default_prompt_template:
                    default_template = await self.storage_service.get_prompt_by_id(
                        config.default_prompt_template
                    )
                    if default_template:
                        formatted_message = await self._format_message_with_template(
                            message, default_template, context
                        )
                    else:
                        formatted_message = message
                else:
                    formatted_message = message
            
            # 构建LangChain消息列表
            langchain_messages = []
            
            # 添加历史消息
            for msg in messages:
                if msg.role == MessageRole.USER:
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.role == MessageRole.ASSISTANT:
                    langchain_messages.append(AIMessage(content=msg.content))
                elif msg.role == MessageRole.SYSTEM:
                    langchain_messages.append(SystemMessage(content=msg.content))
            
            # 添加当前用户消息
            langchain_messages.append(HumanMessage(content=formatted_message))
            
            # 调用LLM
            response = await self.llm.ainvoke(langchain_messages)
            assistant_message = response.content
            
            # 更新会话历史
            now = datetime.now()
            
            # 添加用户消息
            messages.append(ChatMessage(
                role=MessageRole.USER,
                content=message,
                timestamp=now
            ))
            
            # 添加助手回复
            messages.append(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=assistant_message,
                timestamp=now
            ))
            
            # 保存会话历史
            await self.storage_service.save_conversation(
                conversation_id, 
                messages,
                {
                    "prompt_template_used": prompt_template_id,
                    "context": context
                }
            )
            
            # 记录聊天日志
            await self.storage_service.add_log(
                "INFO",
                f"聊天完成 - 会话: {conversation_id}",
                "langchain_service",
                {
                    "conversation_id": conversation_id,
                    "prompt_template": prompt_template_id,
                    "message_length": len(message),
                    "response_length": len(assistant_message)
                }
            )
            
            return {
                "message": assistant_message,
                "conversation_id": conversation_id,
                "timestamp": now,
                "prompt_template_used": prompt_template_id,
                "metadata": {
                    "formatted_message": formatted_message,
                    "context": context
                }
            }
            
        except Exception as e:
            logger.error(f"聊天处理失败: {e}")
            
            # 记录错误日志
            await self.storage_service.add_log(
                "ERROR",
                f"聊天处理失败: {str(e)}",
                "langchain_service",
                {
                    "conversation_id": conversation_id,
                    "error_type": type(e).__name__,
                    "message": message
                }
            )
            
            raise
    
    async def _format_message_with_template(self, 
                                           message: str, 
                                           prompt_template: PromptTemplate,
                                           context: Dict[str, Any] = None) -> str:
        """使用Prompt模板格式化消息"""
        try:
            # 创建LangChain PromptTemplate
            langchain_template = LangChainPromptTemplate(
                template=prompt_template.template,
                input_variables=prompt_template.variables
            )
            
            # 准备变量
            variables = {}
            
            # 添加基本变量
            if "user_input" in prompt_template.variables:
                variables["user_input"] = message
            if "question" in prompt_template.variables:
                variables["question"] = message
            if "message" in prompt_template.variables:
                variables["message"] = message
            
            # 添加上下文变量
            if context:
                for var in prompt_template.variables:
                    if var in context:
                        variables[var] = context[var]
            
            # 检查是否所有必需变量都已提供
            missing_vars = set(prompt_template.variables) - set(variables.keys())
            if missing_vars:
                logger.warning(f"Prompt模板缺少变量: {missing_vars}")
                # 为缺少的变量提供默认值
                for var in missing_vars:
                    variables[var] = f"[{var}]"
            
            # 格式化模板
            formatted_message = langchain_template.format(**variables)
            
            logger.debug(f"Prompt模板格式化完成: {prompt_template.name}")
            return formatted_message
            
        except Exception as e:
            logger.error(f"Prompt模板格式化失败: {e}")
            # 如果格式化失败，返回原始消息
            return message
    
    async def validate_prompt_template(self, template: str, variables: List[str]) -> Dict[str, Any]:
        """验证Prompt模板"""
        try:
            # 创建LangChain PromptTemplate进行验证
            langchain_template = LangChainPromptTemplate(
                template=template,
                input_variables=variables
            )
            
            # 尝试使用测试数据格式化
            test_variables = {var: f"test_{var}" for var in variables}
            test_result = langchain_template.format(**test_variables)
            
            return {
                "valid": True,
                "message": "模板验证成功",
                "test_result": test_result
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"模板验证失败: {str(e)}",
                "error": str(e)
            }
    
    async def update_llm_config(self, config_updates: Dict[str, Any]):
        """更新LLM配置"""
        try:
            # 更新存储的配置
            await self.storage_service.update_config(config_updates)
            
            # 重新初始化LLM
            await self.initialize()
            
            logger.info("LLM配置更新完成")
            
        except Exception as e:
            logger.error(f"LLM配置更新失败: {e}")
            raise