# LangChain Agent 综合演示

基于 `langchain/official-tutorials` 知识点的智能代理应用，整合了提示模板、向量存储、信息提取、Agent 构建等核心功能。

## 🎯 项目特点

本项目是基于 `langchain/official-tutorials` 文件夹下教程知识点的综合实现，包含以下核心功能：

### 核心知识点对应

| 功能模块 | 对应教程文件 | 说明 |
|---------|-------------|------|
| 🔧 提示模板管理 | `1.prompt-templates.py` | ChatPromptTemplate、MessagesPlaceholder |
| 🤖 LLM 模型集成 | `2.simple-llm-app.py` | 模型初始化、流式输出 |
| 📚 向量存储 | `3.vector_stores.py` | 文档分割、嵌入、相似度搜索 |
| 🔍 异步搜索 | `4.similarity_search.py` | 异步相似度搜索、批量处理 |
| 📊 检索器 | `5.retrievers.py` | 批量查询、Runnable 接口 |
| 🏷️ 文本分类 | `6.classification.py` | 结构化输出、分类任务 |
| 📋 信息提取 | `7.extraction.py` | Pydantic 模型、结构化数据提取 |
| 💬 聊天机器人 | `8.chatbot.ipynb` | 对话管理、上下文记忆 |
| 🚀 Agent 构建 | `9.build-an-agent.ipynb` | ReAct Agent、工具集成、记忆管理 |

## 🏗️ 项目结构

```
agent-demo/
├── main.py              # 主程序入口
├── config.py            # 配置管理  
├── tools.py             # 工具定义和数据模型
├── examples.py          # 示例数据和环境配置
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd langchain/agent-demo

# 安装依赖
pip install -r requirements.txt
```

### 2. API 密钥配置

配置所需的 API 密钥（至少需要配置一个 LLM 提供商）：

```bash
# 必需 - LLM 模型 (选择其一)
export DEEPSEEK_API_KEY="sk-your-deepseek-api-key"
export OPENAI_API_KEY="sk-your-openai-api-key"

# 可选 - 嵌入模型 (用于向量搜索)
export SILICONFLOW_API_KEY="sk-your-siliconflow-api-key"

# 可选 - 网络搜索 (用于实时信息查询)
export TAVILY_API_KEY="tvly-your-tavily-api-key"
```

### 3. 运行演示

```bash
# 检查环境配置
python examples.py

# 运行完整演示
python main.py
```

## 🎮 功能演示

### 智能对话 (基于 9.build-an-agent.ipynb)

```python
# 支持多轮对话和工具调用
agent = SmartAgent()
response = await agent.chat("请搜索OpenAI公司的信息")
```

### 向量搜索 (基于 3.vector_stores.py + 4.similarity_search.py)

```python
# 语义相似度搜索
results = agent.vector_search("人工智能公司信息", k=3)
for result in results:
    print(f"相关度: {result['relevance']}")
    print(f"内容: {result['content']}")
```

### 信息提取 (基于 7.extraction.py)

```python
# 结构化信息提取
text = "苹果公司的CEO是蒂姆·库克，总部位于库比蒂诺"
extracted = agent.extract_information(text)
print(f"人员信息: {extracted.people}")
print(f"公司信息: {extracted.companies}")
```

### 批量处理 (基于 5.retrievers.py)

```python
# 批量查询处理
queries = ["OpenAI信息", "机器学习算法", "LangChain框架"]
results = agent.batch_process(queries)
```

## 🔧 配置说明

### 模型配置

在 `config.py` 中可以修改默认模型设置：

```python
class Config:
    # 默认使用 DeepSeek (可改为 "gpt-4o-mini" 等)
    DEFAULT_LLM_MODEL = "deepseek-chat"
    DEFAULT_LLM_PROVIDER = "deepseek"
    
    # 嵌入模型配置
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
```

### 向量存储配置

```python
VECTOR_STORE_CONFIG = {
    "chunk_size": 1000,      # 文档分块大小
    "chunk_overlap": 200,    # 分块重叠
    "similarity_search_k": 3 # 返回结果数量
}
```

## 🛠️ 技术架构

### 核心组件

1. **SmartAgent** - 主要的代理类
   - 集成 LLM 模型、向量存储、搜索工具
   - 支持多轮对话和记忆管理
   - 提供统一的功能接口

2. **VectorStoreManager** - 向量存储管理器
   - 基于 SiliconFlow 嵌入模型
   - 支持文档分割和相似度搜索
   - 内存向量存储 (可扩展为持久化存储)

3. **SearchTools** - 工具集合
   - Tavily 网络搜索
   - 向量搜索工具
   - 文档摘要工具

### 数据模型 (基于 7.extraction.py)

```python
class PersonInfo(BaseModel):
    name: Optional[str] = Field(description="人员姓名")
    role: Optional[str] = Field(description="职位或角色")
    company: Optional[str] = Field(description="所属公司")
    location: Optional[str] = Field(description="所在地点")

class CompanyInfo(BaseModel):
    name: Optional[str] = Field(description="公司名称")
    industry: Optional[str] = Field(description="所属行业")
    location: Optional[str] = Field(description="公司位置")
    description: Optional[str] = Field(description="公司描述")
```

## 📖 教程对应说明

### 1. 提示模板 (1.prompt-templates.py)

```python
# 在 SmartAgent.create_prompt_template() 中实现
prompt_template = ChatPromptTemplate([
    ("system", system_message),
    MessagesPlaceholder("chat_history"),  # 历史对话
    ("human", "{input}")
])
```

### 2. LLM 应用 (2.simple-llm-app.py)

```python
# 在 SmartAgent._initialize_components() 中实现
self.model = init_chat_model(
    self.config.DEFAULT_LLM_MODEL,
    model_provider=self.config.DEFAULT_LLM_PROVIDER
)
```

### 3. 向量存储 (3.vector_stores.py)

```python
# 在 VectorStoreManager 中实现
self.embeddings = SiliconFlowEmbeddings(model=Config.EMBEDDING_MODEL)
self.vector_store = InMemoryVectorStore(self.embeddings)
self.text_splitter = RecursiveCharacterTextSplitter(...)
```

### 4. 相似度搜索 (4.similarity_search.py)

```python
# 在 VectorStoreManager.similarity_search() 中实现
results = self.vector_store.similarity_search(query, k=k)
```

### 5. 检索器 (5.retrievers.py)

```python
# 在 SmartAgent.batch_process() 中实现批量处理
for query in queries:
    results = self.vector_search(query)
    # 处理结果...
```

### 6. 信息提取 (7.extraction.py)

```python
# 在 SmartAgent.extract_information() 中实现
structured_llm = self.model.with_structured_output(schema=ExtractedData)
result = structured_llm.invoke(prompt)
```

### 7. Agent 构建 (9.build-an-agent.ipynb)

```python
# 在 SmartAgent._initialize_components() 中实现
self.agent_executor = create_react_agent(
    self.model, 
    tools, 
    checkpointer=self.memory
)
```

## 🔍 使用示例

### 基础聊天

```python
import asyncio
from main import SmartAgent

async def basic_chat():
    agent = SmartAgent()
    
    # 简单问答
    response = await agent.chat("你好，请介绍一下你的功能")
    print(response)
    
    # 搜索功能 (如果配置了 Tavily)
    response = await agent.chat("请搜索最新的AI技术新闻")
    print(response)

# 运行
asyncio.run(basic_chat())
```

### 向量搜索

```python
from main import SmartAgent

def vector_search_demo():
    agent = SmartAgent()
    
    # 语义搜索
    results = agent.vector_search("人工智能公司")
    
    for result in results:
        print(f"相关度: {result['relevance']}")
        print(f"内容: {result['content'][:200]}...")
        print(f"来源: {result['metadata']['source']}")
        print("-" * 50)

vector_search_demo()
```

### 信息提取

```python
from main import SmartAgent

def extraction_demo():
    agent = SmartAgent()
    
    text = """
    特斯拉公司是一家美国电动汽车制造商，由埃隆·马斯克领导。
    公司总部位于德克萨斯州奥斯汀，专注于电动汽车和清洁能源技术。
    """
    
    result = agent.extract_information(text)
    
    print("提取结果:")
    print(f"人员: {result.people}")
    print(f"公司: {result.companies}")  
    print(f"摘要: {result.summary}")

extraction_demo()
```

## 🚧 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   ❌ LLM模型初始化失败: Invalid API key
   ```
   解决：检查 API 密钥是否正确配置

2. **依赖包缺失**
   ```
   ModuleNotFoundError: No module named 'langchain_siliconflow'
   ```
   解决：运行 `pip install -r requirements.txt`

3. **向量存储初始化失败**
   ```
   ⚠️ 向量存储初始化失败
   ```
   解决：检查 SILICONFLOW_API_KEY 是否配置

### 调试模式

在 `main.py` 中添加详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 扩展功能

### 添加新的工具

在 `tools.py` 中添加自定义工具：

```python
@tool
def custom_tool(query: str) -> str:
    """自定义工具描述"""
    # 工具逻辑
    return result
```

### 支持更多模型

在 `config.py` 中添加新的模型配置：

```python
# 支持更多模型提供商
SUPPORTED_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat", 
    "anthropic": "claude-3-sonnet"
}
```

### 持久化向量存储

替换内存向量存储为持久化存储：

```python
# 使用 Chroma 或 FAISS
from langchain_chroma import Chroma
vector_store = Chroma(embedding_function=embeddings)
```

## 📚 参考资料

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [官方教程代码](../official-tutorials/)
- [LangChain 学习指南](../official-tutorials/LangChain学习指南.md)

## 📄 许可证

MIT License

---

🎉 **恭喜！** 你已经成功创建了一个基于 LangChain 官方教程知识点的综合 Agent 应用。

通过运行 `python main.py` 开始体验吧！