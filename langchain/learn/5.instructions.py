import getpass
import os
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr


class FAQ(BaseModel):
    question: str = Field(..., description="用户问题")
    short_answer: str
    tags: list[str]


parser = PydanticOutputParser(pydantic_object=FAQ)

format_instructions = parser.get_format_instructions()

if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = getpass.getpass(
        "Enter API key for SiliconFlow: "
    )

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")

prompt = ChatPromptTemplate.from_template(
    """你是企业知识库助手。
  {format_instructions}
  问题：{q}"""
)

llm = ChatOpenAI(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    api_key=SecretStr(SILICONFLOW_API_KEY) if SILICONFLOW_API_KEY else None,
)

chain = prompt.partial(format_instructions=format_instructions) | llm | parser

print(chain.invoke({"q": "LangChain 的 LCEL 有何优势？"}))
