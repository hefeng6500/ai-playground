"""
智能 Agent 应用主程序
基于 langchain/official-tutorials 知识点的综合实现

核心功能:
1. 提示模板管理 (基于 1.prompt-templates.py)
2. LLM 模型集成 (基于 2.simple-llm-app.py)  
3. 向量存储和检索 (基于 3.vector_stores.py, 4.similarity_search.py)
4. 信息提取 (基于 7.extraction.py)
5. Agent 构建 (基于 9.build-an-agent.ipynb)
"""

import asyncio
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver  
from langgraph.prebuilt import create_react_agent

from config import Config
from tools import SearchTools, ExtractedData, initialize_vector_store


class SmartAgent:
    """智能代理类 - 整合所有 official-tutorials 知识点"""
    
    def __init__(self):
        self.config = Config()
        self.search_tools = SearchTools()
        self.vector_manager = None
        self.model = None
        self.agent_executor = None
        self.memory = MemorySaver()
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        print("🚀 正在初始化智能代理...")
        
        # 1. 验证API密钥
        api_status = self.config.validate_api_keys()
        print(f"📋 API密钥状态: {api_status}")
        
        # 2. 初始化模型 (参考 2.simple-llm-app.py)
        try:
            self.model = init_chat_model(
                self.config.DEFAULT_LLM_MODEL,
                model_provider=self.config.DEFAULT_LLM_PROVIDER
            )
            print(f"✅ LLM模型初始化成功: {self.config.DEFAULT_LLM_MODEL}")
        except Exception as e:
            print(f"❌ LLM模型初始化失败: {e}")
            return
        
        # 3. 初始化向量存储 (参考 3.vector_stores.py)
        try:
            if api_status.get("siliconflow"):
                self.vector_manager = initialize_vector_store()
            else:
                print("⚠️  SiliconFlow API密钥未配置，跳过向量存储初始化")
        except Exception as e:
            print(f"⚠️  向量存储初始化失败: {e}")
        
        # 4. 创建代理执行器 (参考 9.build-an-agent.ipynb)
        try:
            tools = self.search_tools.get_tools()
            self.agent_executor = create_react_agent(
                self.model, 
                tools, 
                checkpointer=self.memory
            )
            print(f"✅ Agent执行器创建成功，可用工具数量: {len(tools)}")
        except Exception as e:
            print(f"❌ Agent执行器创建失败: {e}")
    
    def create_prompt_template(self, system_message: str = None) -> ChatPromptTemplate:
        """创建提示模板 (参考 1.prompt-templates.py)"""
        if system_message is None:
            system_message = """你是一个智能助手，具备以下能力:
            1. 网络搜索和信息检索
            2. 文档向量搜索  
            3. 信息提取和摘要
            4. 多轮对话记忆
            
            请根据用户的问题，选择合适的工具来提供准确和有用的回答。"""
        
        prompt_template = ChatPromptTemplate([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        return prompt_template
    
    def extract_information(self, text: str) -> ExtractedData:
        """信息提取功能 (参考 7.extraction.py)"""
        if not self.model:
            return ExtractedData()
        
        try:
            # 创建结构化输出模型
            structured_llm = self.model.with_structured_output(schema=ExtractedData)
            
            # 创建提取提示
            extraction_prompt = ChatPromptTemplate([
                ("system", """你是一个专业的信息提取专家。请从给定文本中提取以下信息:
                1. 人员信息: 姓名、职位、公司、地点
                2. 公司信息: 名称、行业、位置、描述  
                3. 内容摘要: 简洁的总结
                
                如果某些信息在文本中不存在，请返回null。"""),
                ("human", "请提取以下文本的信息:\n\n{text}")
            ])
            
            prompt = extraction_prompt.invoke({"text": text})
            result = structured_llm.invoke(prompt)
            
            return result
            
        except Exception as e:
            print(f"信息提取失败: {e}")
            return ExtractedData()
    
    def vector_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """向量搜索功能 (参考 4.similarity_search.py)"""
        if not self.vector_manager:
            return [{"error": "向量存储未初始化"}]
        
        try:
            results = self.vector_manager.similarity_search(query, k=k)
            
            formatted_results = []
            for i, doc in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance": "高" if i == 0 else "中" if i == 1 else "低"
                })
            
            return formatted_results
            
        except Exception as e:
            return [{"error": f"向量搜索失败: {e}"}]
    
    async def chat(self, message: str, session_id: str = "default") -> str:
        """聊天功能 (参考 8.chatbot.ipynb)"""
        if not self.agent_executor:
            return "❌ Agent未正确初始化，请检查配置"
        
        try:
            config = {"configurable": {"thread_id": session_id}}
            
            input_message = {
                "role": "user",
                "content": message
            }
            
            # 流式调用代理
            full_response = ""
            print(f"\n🤖 Agent正在处理: {message}")
            print("="*50)
            
            for step in self.agent_executor.stream(
                {"messages": [input_message]}, 
                config, 
                stream_mode="values"
            ):
                latest_message = step["messages"][-1]
                if hasattr(latest_message, 'content'):
                    full_response = latest_message.content
                    print(f"📝 响应: {latest_message.content}")
            
            return full_response
            
        except Exception as e:
            error_msg = f"❌ 聊天处理失败: {e}"
            print(error_msg)
            return error_msg
    
    def batch_process(self, queries: List[str]) -> List[Dict[str, Any]]:
        """批量处理功能 (参考 5.retrievers.py)"""
        results = []
        
        for i, query in enumerate(queries):
            print(f"\n处理查询 {i+1}/{len(queries)}: {query}")
            
            # 向量搜索
            vector_results = self.vector_search(query)
            
            # 信息提取
            if vector_results and not vector_results[0].get("error"):
                combined_text = "\n".join([r["content"] for r in vector_results[:2]])
                extracted_info = self.extract_information(combined_text)
            else:
                extracted_info = ExtractedData()
            
            results.append({
                "query": query,
                "vector_search": vector_results,
                "extracted_info": extracted_info,
                "timestamp": datetime.now().isoformat()
            })
        
        return results


class AgentDemo:
    """Agent 演示类"""
    
    def __init__(self):
        self.agent = SmartAgent()
    
    def show_capabilities(self):
        """展示 Agent 能力"""
        print("\n🎯 智能代理能力展示")
        print("=" * 40)
        print("1. 💬 智能对话 - 基于提示模板和记忆管理")
        print("2. 🔍 向量搜索 - 语义相似度检索")
        print("3. 📊 信息提取 - 结构化数据提取")
        print("4. 🌐 网络搜索 - 实时信息查询")
        print("5. 📋 批量处理 - 多查询并行处理")
    
    async def demo_chat(self):
        """演示聊天功能"""
        print("\n💬 聊天功能演示")
        print("-" * 30)
        
        demo_questions = [
            "你好，请介绍一下你的功能",
            "请搜索一下OpenAI公司的信息",
            "LangChain框架有什么特点？"
        ]
        
        for question in demo_questions:
            response = await self.agent.chat(question)
            print(f"\n❓ 用户: {question}")
            print(f"🤖 助手: {response}")
    
    def demo_vector_search(self):
        """演示向量搜索功能"""
        print("\n🔍 向量搜索功能演示")
        print("-" * 30)
        
        queries = [
            "人工智能公司信息",
            "机器学习算法",
            "开发框架介绍"
        ]
        
        for query in queries:
            results = self.agent.vector_search(query)
            print(f"\n🔎 查询: {query}")
            for result in results[:2]:  # 只显示前2个结果
                if not result.get("error"):
                    print(f"  📄 {result['content'][:100]}...")
                else:
                    print(f"  ❌ {result['error']}")
    
    def demo_information_extraction(self):
        """演示信息提取功能"""
        print("\n📊 信息提取功能演示")
        print("-" * 30)
        
        sample_text = """
        苹果公司(Apple Inc.)是一家美国科技公司，总部位于加利福尼亚州库比蒂诺。
        现任CEO是蒂姆·库克(Tim Cook)，公司成立于1976年。
        苹果主要从事消费电子产品的设计和制造，包括iPhone、iPad、Mac等产品。
        """
        
        extracted = self.agent.extract_information(sample_text)
        print(f"📝 原文本: {sample_text.strip()}")
        print(f"\n🎯 提取结果:")
        print(f"  👥 人员信息: {extracted.people}")
        print(f"  🏢 公司信息: {extracted.companies}")
        print(f"  📋 摘要: {extracted.summary}")
    
    async def run_demo(self):
        """运行完整演示"""
        print("🚀 启动 LangChain Agent 综合演示")
        print("=" * 50)
        
        # 显示能力
        self.show_capabilities()
        
        # 演示各功能
        try:
            await self.demo_chat()
        except Exception as e:
            print(f"❌ 聊天演示失败: {e}")
        
        self.demo_vector_search()
        self.demo_information_extraction()
        
        print("\n✅ 演示完成!")


# ==================== 主程序入口 ====================

async def main():
    """主程序入口"""
    demo = AgentDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要 Python 3.8 或更高版本")
        sys.exit(1)
    
    # 运行演示
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序已退出")
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")