# LangChain 官方教程学习指南

## 目录
1. [简介](#简介)
2. [基础概念](#基础概念)
3. [核心组件详解](#核心组件详解)
4. [实战应用](#实战应用)
5. [高级功能](#高级功能)
6. [学习路径建议](#学习路径建议)

## 简介

本文档基于 LangChain 官方教程代码，为初学者提供全面的学习指导。LangChain 是一个强大的框架，用于构建基于大语言模型（LLM）的应用程序。

### 什么是 LangChain？
- **定义**：LangChain 是一个用于开发由语言模型驱动的应用程序的框架
- **核心价值**：简化 LLM 应用的开发流程，提供标准化的组件和接口
- **主要功能**：提示管理、链式调用、记忆管理、文档处理、向量存储等

## 基础概念

### 1. 提示模板 (Prompt Templates)
**文件参考**: `1.prompt-templates.py`

提示模板是 LangChain 的基础组件，用于格式化和管理发送给 LLM 的提示。

#### 基本提示模板
```python
from langchain_core.prompts import PromptTemplate

# 创建简单的提示模板
prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
```

#### 聊天提示模板
```python
from langchain_core.prompts import ChatPromptTemplate

# 创建聊天提示模板
prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}"),
])
```

#### 消息占位符
```python
from langchain_core.prompts import MessagesPlaceholder

# 用于插入历史对话消息
prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"), 
    MessagesPlaceholder("msgs")
])
```

**关键概念**：
- **变量替换**：使用 `{variable}` 语法进行动态内容替换
- **角色区分**：system、user、assistant 等不同角色的消息
- **历史管理**：MessagesPlaceholder 用于管理对话历史

### 2. 简单 LLM 应用
**文件参考**: `2.simple-llm-app.py`

展示如何创建基本的 LLM 应用，包括模型初始化、消息处理和流式输出。

#### 模型初始化
```python
from langchain.chat_models import init_chat_model

# 初始化聊天模型
model = init_chat_model("deepseek-chat", model_provider="deepseek")
```

#### 消息处理
```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("Hello World!"),
]

response = model.invoke(messages)
```

#### 流式输出
```python
# 实时获取模型输出
for token in model.stream(messages):
    print(token.content, end="|")
```

**关键概念**：
- **模型提供商**：支持多种 LLM 提供商（OpenAI、DeepSeek 等）
- **消息类型**：SystemMessage、HumanMessage、AIMessage
- **调用方式**：同步调用 (invoke) 和流式调用 (stream)

## 核心组件详解

### 3. 向量存储 (Vector Stores)
**文件参考**: `3.vector_stores.py`

向量存储是实现语义搜索和 RAG（检索增强生成）的核心组件。

#### 文档处理流程
```python
# 1. 文档加载
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(file_path)
docs = loader.load()

# 2. 文档分割
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# 3. 向量化
from langchain_siliconflow import SiliconFlowEmbeddings
embeddings = SiliconFlowEmbeddings(model="Qwen/Qwen3-Embedding-8B")

# 4. 存储到向量数据库
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)
```

**关键概念**：
- **文档分割**：将长文档分割成小块，便于检索
- **嵌入模型**：将文本转换为向量表示
- **相似度搜索**：基于向量相似度查找相关文档
- **chunk_size**：每个文档块的大小
- **chunk_overlap**：文档块之间的重叠部分

### 4. 相似度搜索
**文件参考**: `4.similarity_search.py`

展示如何进行异步相似度搜索，提高查询效率。

```python
# 异步查询函数
async def query_vector_store(query: str, k: int = 3):
    results = await vector_store.asimilarity_search(query, k=k)
    return results

# 使用示例
query = "When was Nike incorporated?"
results = await query_vector_store(query)
```

**关键概念**：
- **异步处理**：提高查询性能
- **Top-K 检索**：返回最相似的 K 个结果
- **查询优化**：合理设置检索参数

### 5. 检索器 (Retrievers)
**文件参考**: `5.retrievers.py`

检索器是对向量存储的高级封装，支持批量查询和链式操作。

```python
from langchain_core.runnables import RunnableLambda

# 创建检索器
retriever = RunnableLambda(lambda q: vector_store.similarity_search(q, k=1))

# 批量查询
results = retriever.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
])
```

**关键概念**：
- **Runnable 接口**：LangChain 的标准化接口
- **批量处理**：同时处理多个查询
- **链式操作**：可以与其他组件组合使用

### 6. 文本分类
**文件参考**: `6.classification.py`

使用结构化输出进行文本分类和情感分析。

```python
from pydantic import BaseModel, Field

# 定义分类结构
class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")

# 创建结构化输出模型
structured_llm = llm.with_structured_output(Classification)
```

**关键概念**：
- **Pydantic 模型**：定义结构化数据格式
- **结构化输出**：确保 LLM 输出符合预定义格式
- **字段描述**：帮助 LLM 理解每个字段的含义

### 7. 信息提取
**文件参考**: `7.extraction.py`

从非结构化文本中提取结构化信息。

```python
class Person(BaseModel):
    """Information about a person."""
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )

# 多实体提取
class Data(BaseModel):
    people: List[Person]
```

**关键概念**：
- **实体提取**：识别文本中的特定实体
- **多实体处理**：同时提取多个实体
- **Few-shot 学习**：通过示例提高提取质量

## 实战应用

### 8. 聊天机器人
**文件参考**: `8.chatbot.ipynb`

构建具有记忆功能的聊天机器人。

#### 基本聊天
```python
# 无状态聊天
response = model.invoke([HumanMessage(content="Hi! I'm Bob")])

# 有状态聊天（手动管理历史）
response = model.invoke([
    HumanMessage(content="Hi! I'm Bob"),
    AIMessage(content="Hello Bob! How can I assist you today?"),
    HumanMessage(content="What's my name?"),
])
```

#### 消息持久化
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState

# 创建带记忆的工作流
workflow = StateGraph(state_schema=MessagesState)
memory = MemorySaver()
```

**关键概念**：
- **对话状态管理**：维护对话历史
- **消息持久化**：保存对话记录
- **状态图**：管理复杂的对话流程

### 9. 智能代理
**文件参考**: `9.build-an-agent.ipynb`

构建能够使用工具的智能代理。

**关键概念**：
- **工具调用**：代理可以调用外部工具
- **决策制定**：代理根据用户输入决定使用哪些工具
- **多步推理**：代理可以进行多步骤的复杂任务

### 10-14. 高级应用

#### RAG 应用 (`10.rag-01.ipynb`, `11.rag-02.ipynb`)
- **检索增强生成**：结合外部知识库的问答系统
- **多步检索**：复杂查询的分步处理
- **上下文管理**：有效管理检索到的信息

#### SQL 数据查询 (`12.sql-data.ipynb`)
- **自然语言转 SQL**：将用户问题转换为 SQL 查询
- **数据库集成**：与关系数据库的无缝集成
- **结果解释**：将查询结果转换为自然语言回答

#### 文本摘要 (`13.summarize-text.ipynb`)
- **长文本处理**：处理超长文档的摘要
- **分层摘要**：先分段摘要，再整体摘要
- **摘要质量控制**：确保摘要的准确性和完整性

#### 图数据库查询 (`14.graph-database.ipynb`)
- **知识图谱查询**：基于图结构的知识检索
- **关系推理**：理解实体间的复杂关系
- **图查询语言**：生成 Cypher 等图查询语言

## 高级功能

### 链式操作 (Chains)
```python
# 创建处理链
chain = prompt_template | model | output_parser
result = chain.invoke({"input": "用户输入"})
```

### 内存管理 (Memory)
```python
# 对话记忆
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
```

### 输出解析 (Output Parsers)
```python
# 结构化输出解析
from langchain.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=YourModel)
```

## 学习路径建议

### 初学者路径
1. **基础概念** (第1-2周)
   - 学习提示模板 (`1.prompt-templates.py`)
   - 创建简单 LLM 应用 (`2.simple-llm-app.py`)
   - 理解消息类型和模型调用

2. **核心组件** (第3-4周)
   - 掌握向量存储 (`3.vector_stores.py`)
   - 学习相似度搜索 (`4.similarity_search.py`)
   - 了解检索器使用 (`5.retrievers.py`)

3. **结构化处理** (第5-6周)
   - 文本分类 (`6.classification.py`)
   - 信息提取 (`7.extraction.py`)
   - 结构化输出的应用

### 进阶路径
4. **应用开发** (第7-10周)
   - 聊天机器人开发 (`8.chatbot.ipynb`)
   - 智能代理构建 (`9.build-an-agent.ipynb`)
   - RAG 系统实现 (`10-11.rag-*.ipynb`)

5. **专业应用** (第11-12周)
   - SQL 查询系统 (`12.sql-data.ipynb`)
   - 文本摘要系统 (`13.summarize-text.ipynb`)
   - 图数据库应用 (`14.graph-database.ipynb`)

### 实践建议

1. **环境准备**
   ```bash
   pip install langchain langchain-community langchain-core
   pip install langchain-openai  # 或其他模型提供商
   ```

2. **API 密钥配置**
   - 设置环境变量存储 API 密钥
   - 使用 `.env` 文件管理敏感信息

3. **逐步实践**
   - 从简单示例开始
   - 逐步增加复杂度
   - 多做实验和调试

4. **社区资源**
   - 官方文档：https://python.langchain.com/
   - GitHub 仓库：https://github.com/langchain-ai/langchain
   - 社区论坛和讨论组

### 常见问题解答

**Q: 如何选择合适的嵌入模型？**
A: 根据你的语言需求和性能要求选择。中文应用推荐使用 Qwen 系列，英文可以使用 OpenAI 的 text-embedding-ada-002。

**Q: 向量数据库如何选择？**
A: 开发阶段可以使用 InMemoryVectorStore，生产环境推荐 Chroma、Pinecone 或 Weaviate。

**Q: 如何优化 RAG 系统的性能？**
A: 关键在于文档分割策略、嵌入模型选择、检索参数调优和重排序机制。

**Q: LangChain 与其他框架的区别？**
A: LangChain 提供了更完整的生态系统和标准化接口，特别适合快速原型开发和复杂应用构建。

## 总结

LangChain 是一个功能强大的 LLM 应用开发框架，通过本指南的学习，你应该能够：

1. 理解 LangChain 的核心概念和组件
2. 掌握基本的应用开发技能
3. 构建实用的 LLM 应用
4. 解决常见的开发问题

记住，学习 LangChain 最好的方法是实践。建议你跟着教程代码一步步操作，并尝试修改参数和逻辑，观察不同的效果。随着经验的积累，你将能够构建更复杂和实用的 LLM 应用。

---

*本文档基于 LangChain 官方教程整理，持续更新中。如有问题或建议，欢迎反馈。*