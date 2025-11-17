# RAG 应用

一个完整的检索增强生成（RAG）应用，基于 LangChain 构建。

## 功能特性

- 📄 **多格式文档支持**：支持 TXT、PDF、Markdown 等格式
- 🔍 **智能检索**：基于向量相似度的语义检索
- 🧠 **增强生成**：结合检索结果生成准确回答
- 💾 **持久化存储**：向量存储可保存和加载
- 💬 **交互式问答**：命令行交互界面

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置 API Key（选择一种）：
```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# 或 SiliconFlow
export SILICONFLOW_API_KEY="your-api-key"
```

## 使用方法

### 基本使用

```bash
python main.py
```

首次运行时会提示输入文档路径，之后可以直接进行问答。

### 编程方式使用

```python
from main import RAGApplication
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 初始化
rag = RAGApplication(
    embeddings_model=OpenAIEmbeddings(),
    llm_model=ChatOpenAI(model="gpt-3.5-turbo"),
)

# 加载文档
documents = rag.load_documents("./documents")

# 构建向量存储
rag.build_vector_store(documents)

# 查询
answer = rag.query("你的问题是什么？")
print(answer)
```

## 项目结构

```
rag/
├── main.py              # 主应用文件
├── requirements.txt     # 依赖列表
├── README.md           # 说明文档
└── vector_store/       # 向量存储目录（自动创建）
```

## 支持的文档格式

- `.txt` - 纯文本文件
- `.pdf` - PDF 文档
- `.md` - Markdown 文件

## 配置选项

在 `RAGApplication` 初始化时可以配置：

- `chunk_size`: 文档分块大小（默认 1000）
- `chunk_overlap`: 分块重叠大小（默认 200）
- `vector_store_path`: 向量存储路径（默认 "./vector_store"）

## 注意事项

- 首次使用需要提供文档路径构建知识库
- 向量存储会自动保存，下次运行可直接使用
- 如需更新知识库，可以删除 `vector_store` 目录后重新运行

