import getpass
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
from dotenv import load_dotenv

if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a concise assistant."), ("user", "{input}")]
)

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=SecretStr(DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None,
)


chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke({"input": "你是谁？"}))
