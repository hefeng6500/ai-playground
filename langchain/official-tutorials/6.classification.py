from pydantic import BaseModel, Field

# --- 1) 选择模型（两种示例：Google Gemini via init_chat_model, 或 OpenAI via ChatOpenAI） ---
# Option A: Google Gemini via init_chat_model (示例来自官方)
from langchain.chat_models import init_chat_model


llm = init_chat_model("deepseek-chat", model_provider="deepseek")

# Option B: OpenAI Chat model (如果使用 OpenAI，取消下面注释)
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# --- 2) Prompt 模板（把要标注的文本塞进 {input}） ---
from langchain_core.prompts import ChatPromptTemplate

tagging_prompt = ChatPromptTemplate.from_template(
    """Extract the desired information from the following passage.
Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


# --- 3) 定义结构化输出的 Pydantic schema ---
class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


# --- 4) 将模型包装为“结构化输出”模型（关键 API） ---
structured_llm = llm.with_structured_output(Classification)

# --- 5) 调用并查看结果（返回的是 Pydantic 对象，可用 .model_dump() 转 dict） ---
inp = "操你妈逼！"
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

print("Pydantic model:", response)  # Classification(sentiment='positive', ...)
print("Dict output:", response.model_dump())  # {'sentiment': 'positive', ...}
