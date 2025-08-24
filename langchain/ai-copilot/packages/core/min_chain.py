from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

if not os.environ.get("DEEPSEEK_API_KEY"):
  os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

API_KEY = os.environ.get("DEEPSEEK_API_KEY")

prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a concise assistant."),
  ("user", "{input}")
])

llm = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek 模型
    base_url="https://api.deepseek.com",  # DeepSeek API 端点
    api_key=API_KEY  # 请替换为您的 DeepSeek API 密钥
)
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke({"input": "用一句话解释 LCEL 是什么？"}))

