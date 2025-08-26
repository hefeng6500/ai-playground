import getpass
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from dotenv import load_dotenv


if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = getpass.getpass(
        "Enter API key for SiliconFlow: "
    )

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a concise assistant."), ("user", "{input}")]
)

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     base_url="https://api.deepseek.com",
#     api_key=API_KEY
# )

llm = ChatOpenAI(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    api_key=SecretStr(SILICONFLOW_API_KEY) if SILICONFLOW_API_KEY else None,
)


chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke({"input": "你是谁？"}))
