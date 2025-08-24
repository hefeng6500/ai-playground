from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

if not os.environ.get("DEEPSEEK_API_KEY"):
  os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a concise assistant."),
  ("user", "{input}")
])

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     base_url="https://api.deepseek.com",
#     api_key=API_KEY
# )

llm = ChatOpenAI(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    api_key=SILICONFLOW_API_KEY
)


chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke({"input": "你是谁？"}))

