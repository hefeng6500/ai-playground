import os
import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_siliconflow import SiliconFlowEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda


file_path = "./example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# -----------------------------
# 1. 假设 docs 是你要查询的文档列表
# docs = [Document(page_content="你的文本内容1"), Document(page_content="文本2"), ...]
# -----------------------------

# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# 3. 初始化 SiliconFlow Embeddings
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

embeddings = SiliconFlowEmbeddings(
    model="Qwen/Qwen3-Embedding-8B", siliconflow_api_key=SILICONFLOW_API_KEY
)

# 4. 建立向量存储
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)


# 5. 定义异步查询函数
async def query_vector_store(query: str, k: int = 3):
    results = await vector_store.asimilarity_search(query, k=k)
    return results


# 包装成 Runnable
retriever = RunnableLambda(lambda q: vector_store.similarity_search(q, k=1))

# 批量运行
results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]
)

for i, r in enumerate(results):
    print(f"Query {i+1}:", r[0].page_content)
