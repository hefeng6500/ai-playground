"""
工具定义模块 - 基于 official-tutorials 知识点
整合搜索工具、向量存储等功能
"""
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_siliconflow import SiliconFlowEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from config import Config


# ==================== 数据模型定义 ====================
# 参考 7.extraction.py 的 Pydantic 模型定义

class PersonInfo(BaseModel):
    """人员信息提取模型"""
    name: Optional[str] = Field(default=None, description="人员姓名")
    role: Optional[str] = Field(default=None, description="职位或角色")
    company: Optional[str] = Field(default=None, description="所属公司")
    location: Optional[str] = Field(default=None, description="所在地点")

class CompanyInfo(BaseModel):
    """公司信息提取模型"""
    name: Optional[str] = Field(default=None, description="公司名称")
    industry: Optional[str] = Field(default=None, description="所属行业")
    location: Optional[str] = Field(default=None, description="公司位置")
    description: Optional[str] = Field(default=None, description="公司描述")

class ExtractedData(BaseModel):
    """综合信息提取结果"""
    people: List[PersonInfo] = Field(default_factory=list, description="人员信息列表")
    companies: List[CompanyInfo] = Field(default_factory=list, description="公司信息列表")
    summary: Optional[str] = Field(default=None, description="内容摘要")


# ==================== 工具类定义 ====================

class VectorStoreManager:
    """向量存储管理器 - 基于 3.vector_stores.py"""
    
    def __init__(self):
        self.embeddings = SiliconFlowEmbeddings(
            model=Config.EMBEDDING_MODEL,
            siliconflow_api_key=Config.SILICONFLOW_API_KEY
        )
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.VECTOR_STORE_CONFIG["chunk_size"],
            chunk_overlap=Config.VECTOR_STORE_CONFIG["chunk_overlap"],
            add_start_index=True
        )
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """添加文档到向量存储"""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {"source": f"doc_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))
        
        # 分割文档
        splits = self.text_splitter.split_documents(documents)
        
        # 添加到向量存储
        ids = self.vector_store.add_documents(documents=splits)
        return ids
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """相似度搜索 - 参考 4.similarity_search.py"""
        if k is None:
            k = Config.VECTOR_STORE_CONFIG["similarity_search_k"]
        
        results = self.vector_store.similarity_search(query, k=k)
        return results


class SearchTools:
    """搜索工具集 - 基于 9.build-an-agent.ipynb"""
    
    def __init__(self):
        self.tavily_search = TavilySearch(
            max_results=Config.TAVILY_MAX_RESULTS
        ) if Config.TAVILY_API_KEY else None
        self.vector_manager = VectorStoreManager()
    
    def get_tools(self) -> List:
        """获取所有可用工具"""
        tools = []
        
        if self.tavily_search:
            tools.append(self.tavily_search)
        
        # 添加自定义工具
        tools.extend([
            self.vector_search_tool,
            self.document_summarizer_tool
        ])
        
        return tools
    
    @tool
    def vector_search_tool(self, query: str) -> str:
        """向量搜索工具"""
        results = self.vector_manager.similarity_search(query)
        if not results:
            return "未找到相关文档"
        
        context = "\n\n".join([
            f"文档{i+1}: {doc.page_content[:200]}..."
            for i, doc in enumerate(results[:2])
        ])
        return f"搜索结果:\n{context}"
    
    @tool  
    def document_summarizer_tool(self, text: str) -> str:
        """文档摘要工具"""
        if len(text) > 500:
            return f"文档摘要: {text[:500]}... (文档较长，已截取前500字符)"
        return f"文档内容: {text}"


# ==================== 辅助函数 ====================

def create_sample_documents() -> List[str]:
    """创建示例文档 - 参考官方教程的示例数据"""
    return [
        """
        OpenAI 是一家专注于人工智能研究的公司，成立于2015年。
        公司的CEO是Sam Altman，总部位于美国旧金山。
        OpenAI 开发了GPT系列模型，包括ChatGPT和GPT-4等知名产品。
        """,
        """
        DeepSeek 是一家中国的AI公司，专注于大语言模型的研发。
        公司开发了DeepSeek-Chat等对话模型，在编程和推理任务上表现优秀。
        DeepSeek的模型具有强大的代码生成和数学推理能力。
        """,
        """
        LangChain 是一个用于构建基于大语言模型应用的框架。
        它提供了提示模板、向量存储、代理等核心组件。
        开发者可以使用LangChain快速构建RAG系统和AI代理应用。
        """,
        """
        机器学习是人工智能的一个分支，通过数据训练模型来进行预测。
        常见的算法包括线性回归、决策树、支持向量机等。
        深度学习作为机器学习的子集，使用神经网络进行复杂的模式识别。
        """
    ]


def initialize_vector_store() -> VectorStoreManager:
    """初始化向量存储并添加示例数据"""
    manager = VectorStoreManager()
    
    # 添加示例文档
    sample_texts = create_sample_documents()
    sample_metadata = [
        {"source": "openai_info", "category": "company"},
        {"source": "deepseek_info", "category": "company"}, 
        {"source": "langchain_info", "category": "framework"},
        {"source": "ml_info", "category": "technology"}
    ]
    
    manager.add_documents(sample_texts, sample_metadata)
    print("✅ 向量存储初始化完成，已添加示例文档")
    
    return manager