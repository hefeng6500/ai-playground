# retrievers(检索)

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# print(documents)

from langchain_community.document_loaders import PyPDFLoader

file_path = "./example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# print(len(docs))

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

# Splitting 拆分

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))

# Embeddings

import getpass
import os


SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

from langchain_siliconflow import SiliconFlowEmbeddings

embeddings = SiliconFlowEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    siliconflow_api_key=SILICONFLOW_API_KEY,  # 如果环境变量已设置可以省略
)

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)


assert len(vector_1) == len(vector_2)

"""
为什么需要这样做？ assert len(vector_1) == len(vector_2)

Embedding 向量必须同维度

无论你有多少段文本，embedding 模型在同一次调用里应该返回 相同长度的向量。

举例：Qwen/Qwen3-Embedding-8B 输出可能是固定 1024 维。

如果向量长度不一致，后续就无法进行 余弦相似度、点积、向量检索 等数学操作。

提前发现异常

assert 是一种“断言”：如果条件为假，程序会立刻抛出 AssertionError，避免带着错误结果继续执行。

比如，如果 API 调用异常（返回不完整向量），或者数据传错导致输出格式不符，这行代码能第一时间拦截。

开发时的防御性编程

在正常情况下，同一个 embedding 模型生成的向量长度一定是相同的，所以这个断言通常总是通过。

写下它的意义是：提醒自己和别人，这里隐含了“所有向量维度必须一致”的假设。
"""


# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

# Vector stores 向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

# print(ids)

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])
