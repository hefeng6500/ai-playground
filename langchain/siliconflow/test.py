import getpass
import os
from langchain_siliconflow import ChatSiliconFlow

if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = getpass.getpass(
        "Enter API key for SiliconFlow: "
    )

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

llm = ChatSiliconFlow(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    siliconflow_api_key=SILICONFLOW_API_KEY,  # 如果环境变量已设置可以省略
)

print(llm.invoke("你好！"))
