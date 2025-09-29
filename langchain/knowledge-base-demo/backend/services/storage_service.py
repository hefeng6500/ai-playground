import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from models import PromptTemplate, SystemConfig, LogEntry, ConversationHistory, ChatMessage

class StorageService:
    """本地JSON文件存储服务"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.prompts_file = self.data_dir / "prompts.json"
        self.config_file = self.data_dir / "config.json"
        self.logs_file = self.data_dir / "logs.json"
        self.conversations_file = self.data_dir / "conversations.json"
        
    async def initialize(self):
        """初始化存储服务"""
        logger.info("初始化存储服务...")
        
        # 创建数据目录
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化各个数据文件
        await self._init_prompts_file()
        await self._init_config_file()
        await self._init_logs_file()
        await self._init_conversations_file()
        
        logger.info("存储服务初始化完成")
    
    async def close(self):
        """关闭存储服务"""
        logger.info("存储服务关闭")
    
    # Prompt模板相关方法
    async def get_all_prompts(self) -> List[PromptTemplate]:
        """获取所有Prompt模板"""
        data = await self._read_json_file(self.prompts_file)
        return [PromptTemplate(**prompt) for prompt in data.get("prompts", [])]
    
    async def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptTemplate]:
        """根据ID获取Prompt模板"""
        prompts = await self.get_all_prompts()
        for prompt in prompts:
            if prompt.id == prompt_id:
                return prompt
        return None
    
    async def create_prompt(self, prompt_data: Dict[str, Any]) -> PromptTemplate:
        """创建新的Prompt模板"""
        prompt_id = str(uuid.uuid4())
        now = datetime.now()
        
        prompt = PromptTemplate(
            id=prompt_id,
            created_at=now,
            updated_at=now,
            **prompt_data
        )
        
        data = await self._read_json_file(self.prompts_file)
        data["prompts"].append(prompt.model_dump())
        await self._write_json_file(self.prompts_file, data)
        
        logger.info(f"创建Prompt模板: {prompt.name} (ID: {prompt_id})")
        return prompt
    
    async def update_prompt(self, prompt_id: str, update_data: Dict[str, Any]) -> Optional[PromptTemplate]:
        """更新Prompt模板"""
        data = await self._read_json_file(self.prompts_file)
        prompts = data.get("prompts", [])
        
        for i, prompt_dict in enumerate(prompts):
            if prompt_dict["id"] == prompt_id:
                # 更新字段
                for key, value in update_data.items():
                    if value is not None:
                        prompt_dict[key] = value
                prompt_dict["updated_at"] = datetime.now().isoformat()
                
                await self._write_json_file(self.prompts_file, data)
                
                updated_prompt = PromptTemplate(**prompt_dict)
                logger.info(f"更新Prompt模板: {updated_prompt.name} (ID: {prompt_id})")
                return updated_prompt
        
        return None
    
    async def delete_prompt(self, prompt_id: str) -> bool:
        """删除Prompt模板"""
        data = await self._read_json_file(self.prompts_file)
        prompts = data.get("prompts", [])
        
        original_length = len(prompts)
        data["prompts"] = [p for p in prompts if p["id"] != prompt_id]
        
        if len(data["prompts"]) < original_length:
            await self._write_json_file(self.prompts_file, data)
            logger.info(f"删除Prompt模板 (ID: {prompt_id})")
            return True
        
        return False
    
    # 系统配置相关方法
    async def get_config(self) -> SystemConfig:
        """获取系统配置"""
        data = await self._read_json_file(self.config_file)
        return SystemConfig(**data.get("config", {}))
    
    async def update_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """更新系统配置"""
        data = await self._read_json_file(self.config_file)
        current_config = data.get("config", {})
        
        # 更新配置
        for key, value in config_data.items():
            if value is not None:
                current_config[key] = value
        
        current_config["updated_at"] = datetime.now().isoformat()
        data["config"] = current_config
        
        await self._write_json_file(self.config_file, data)
        
        config = SystemConfig(**current_config)
        logger.info("系统配置已更新")
        return config
    
    # 日志相关方法
    async def add_log(self, level: str, message: str, source: str = None, metadata: Dict[str, Any] = None):
        """添加日志条目"""
        log_id = str(uuid.uuid4())
        log_entry = LogEntry(
            id=log_id,
            level=level,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata
        )
        
        data = await self._read_json_file(self.logs_file)
        data["logs"].append(log_entry.dict())
        
        # 保持最近1000条日志
        if len(data["logs"]) > 1000:
            data["logs"] = data["logs"][-1000:]
        
        await self._write_json_file(self.logs_file, data)
    
    async def get_logs(self, limit: int = 100, level: str = None) -> List[LogEntry]:
        """获取日志列表"""
        data = await self._read_json_file(self.logs_file)
        logs = data.get("logs", [])
        
        # 按级别过滤
        if level:
            logs = [log for log in logs if log.get("level") == level]
        
        # 按时间倒序排列并限制数量
        logs = sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
        logs = logs[:limit]
        
        return [LogEntry(**log) for log in logs]
    
    # 会话历史相关方法
    async def save_conversation(self, conversation_id: str, messages: List[ChatMessage], metadata: Dict[str, Any] = None):
        """保存会话历史"""
        data = await self._read_json_file(self.conversations_file)
        conversations = data.get("conversations", {})
        
        now = datetime.now()
        conversation = ConversationHistory(
            conversation_id=conversation_id,
            messages=messages,
            created_at=conversations.get(conversation_id, {}).get("created_at", now),
            updated_at=now,
            metadata=metadata
        )
        
        conversations[conversation_id] = conversation.dict()
        data["conversations"] = conversations
        
        await self._write_json_file(self.conversations_file, data)
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """获取会话历史"""
        data = await self._read_json_file(self.conversations_file)
        conversations = data.get("conversations", {})
        
        if conversation_id in conversations:
            return ConversationHistory(**conversations[conversation_id])
        
        return None
    
    async def get_all_conversations(self, limit: int = 50) -> List[ConversationHistory]:
        """获取所有会话历史"""
        data = await self._read_json_file(self.conversations_file)
        conversations = data.get("conversations", {})
        
        # 按更新时间倒序排列
        sorted_conversations = sorted(
            conversations.values(),
            key=lambda x: x.get("updated_at", ""),
            reverse=True
        )
        
        return [ConversationHistory(**conv) for conv in sorted_conversations[:limit]]
    
    # 私有辅助方法
    async def _read_json_file(self, file_path: Path) -> Dict[str, Any]:
        """读取JSON文件"""
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return {}
    
    async def _write_json_file(self, file_path: Path, data: Dict[str, Any]):
        """写入JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except IOError as e:
            logger.error(f"写入文件失败 {file_path}: {e}")
            raise
    
    async def _init_prompts_file(self):
        """初始化Prompt模板文件"""
        if not self.prompts_file.exists():
            default_prompts = {
                "prompts": [
                    {
                        "id": "default-chat",
                        "name": "默认聊天",
                        "description": "通用聊天助手模板",
                        "template": "你是一个有用的AI助手。请根据用户的问题提供准确、有帮助的回答。\n\n用户问题: {user_input}",
                        "variables": ["user_input"],
                        "category": "通用",
                        "is_active": True,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    },
                    {
                        "id": "knowledge-qa",
                        "name": "知识问答",
                        "description": "专业知识问答模板",
                        "template": "你是一个专业的知识问答助手。请基于你的知识库，为用户提供准确、详细的答案。如果不确定答案，请诚实说明。\n\n问题: {question}\n\n请提供详细的回答:",
                        "variables": ["question"],
                        "category": "知识库",
                        "is_active": True,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                ]
            }
            await self._write_json_file(self.prompts_file, default_prompts)
    
    async def _init_config_file(self):
        """初始化配置文件"""
        if not self.config_file.exists():
            default_config = {
                "config": {
                    "openai_model": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "default_prompt_template": "default-chat",
                    "enable_logging": True,
                    "log_level": "INFO",
                    "updated_at": datetime.now().isoformat()
                }
            }
            await self._write_json_file(self.config_file, default_config)
    
    async def _init_logs_file(self):
        """初始化日志文件"""
        if not self.logs_file.exists():
            default_logs = {"logs": []}
            await self._write_json_file(self.logs_file, default_logs)
    
    async def _init_conversations_file(self):
        """初始化会话文件"""
        if not self.conversations_file.exists():
            default_conversations = {"conversations": {}}
            await self._write_json_file(self.conversations_file, default_conversations)