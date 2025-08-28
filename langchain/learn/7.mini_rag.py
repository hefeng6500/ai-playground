import getpass
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_siliconflow import SiliconFlowEmbeddings
from langchain_community.vectorstores import FAISS

if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = getpass.getpass(
        "Enter API key for SiliconFlow: "
    )

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

raw = "LangChain 简化 LLM 应用生命周期……（你的企业文档/Markdown）"

docs = [Document(page_content=raw)]

splits = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100
).split_documents(docs)


embeddings = SiliconFlowEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    siliconflow_api_key=SILICONFLOW_API_KEY,  # 如果环境变量已设置可以省略
)

vs = FAISS.from_documents(splits, embeddings)
retriever = vs.as_retriever(search_type="mmr", k=4)
