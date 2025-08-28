import getpass
import os
from langchain_siliconflow import SiliconFlowEmbeddings

if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = getpass.getpass(
        "Enter API key for SiliconFlow: "
    )

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

embeddings = SiliconFlowEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    siliconflow_api_key=SILICONFLOW_API_KEY,  # 如果环境变量已设置可以省略
)

print(len(embeddings.embed_query("What is the meaning of life?")))
