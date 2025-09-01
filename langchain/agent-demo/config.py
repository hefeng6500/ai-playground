"""
配置文件 - 基于 official-tutorials 的知识点
包含模型配置、API密钥管理等
"""
import os
from typing import Dict, Any

class Config:
    """配置管理类"""
    
    # API 密钥配置
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    
    # 模型配置
    DEFAULT_LLM_MODEL = "deepseek-chat"
    DEFAULT_LLM_PROVIDER = "deepseek"
    
    # 嵌入模型配置 (参考 3.vector_stores.py)
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
    
    # 向量存储配置
    VECTOR_STORE_CONFIG = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "similarity_search_k": 3
    }
    
    # 搜索工具配置
    TAVILY_MAX_RESULTS = 2
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "model": cls.DEFAULT_LLM_MODEL,
            "model_provider": cls.DEFAULT_LLM_PROVIDER
        }
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """验证API密钥是否配置"""
        return {
            "deepseek": bool(cls.DEEPSEEK_API_KEY),
            "siliconflow": bool(cls.SILICONFLOW_API_KEY),
            "openai": bool(cls.OPENAI_API_KEY),
            "tavily": bool(cls.TAVILY_API_KEY)
        }