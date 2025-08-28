# build_extraction_chain_full_example.py
# 说明：此示例忠实整合官方页面 "Build an Extraction Chain" 的所有代码片段与示例逻辑（仅作演示）。
# 运行前请确保：pip install --upgrade langchain-core  且 使用支持 tool-calling 的模型/SDK。
# 推荐在 Jupyter Notebook 中交互运行，或在脚本中设置环境变量。

import os
import getpass
from typing import Optional, List

from pydantic import BaseModel, Field


# ---------------------------
# The Schema (页面 code block)
# ---------------------------
class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.
    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


# ---------------------------
# Prompt template (页面 code block)
# ---------------------------
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

# ---------------------------
# Model init + structured output (页面 code block)
# ---------------------------
# Install note shown on page:
# pip install -qU "langchain[google-genai]"


from langchain.chat_models import init_chat_model

# 与页面一致，示例使用 Google Gemini 初始化
llm = init_chat_model("deepseek-chat", model_provider="deepseek")

# 把模型包装为“结构化输出”，schema 传入 Pydantic model
structured_llm = llm.with_structured_output(schema=Person)

# 测试示例（页面中的演示）
text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
# 调用并打印结果（页面显示的行为：模型会转换 feet -> meters）
resp = structured_llm.invoke(prompt)
print(
    "示例输出（Alan Smith）:", resp
)  # 页面显示: Person(name='Alan Smith', hair_color='blond', height_in_meters='1.83')


# ---------------------------
# Multiple Entities (页面 code block)
# ---------------------------
class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]


# 把 schema 换成 Data（以提取多个人）
structured_llm_multi = llm.with_structured_output(schema=Data)
text2 = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt2 = prompt_template.invoke({"text": text2})
resp2 = structured_llm_multi.invoke(prompt2)
print("示例输出（多实体）:", resp2)

# 页面示例输出:
# Data(people=[Person(name='Jeff', hair_color='black', height_in_meters='1.83'), Person(name='Anna', hair_color='black', height_in_meters=None)])

# ---------------------------
# Reference examples & tool_example_to_messages (页面 code block)
# ---------------------------
from langchain_core.utils.function_calling import tool_example_to_messages

examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        Data(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]

messages = []
for txt, tool_call in examples:
    if tool_call.people:
        # This final message is optional for some providers
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    # 这一步会把 Pydantic 工具调用示例转换为 provider-specific 的 message 序列
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

# 查看生成的 messages（页面用 message.pretty_print() 展示）
for message in messages:
    # message.pretty_print() 在页面中演示，某些 provider 的 message object 支持该方法
    try:
        message.pretty_print()
    except Exception:
        # 兜底打印
        print(message)

# ---------------------------
# 比较有/无 examples 的行为（页面演示）
# ---------------------------
message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon.",
}

# 1) 直接调用（没有示例）
structured_llm_data = llm.with_structured_output(schema=Data)
res_no_ex = structured_llm_data.invoke([message_no_extraction])
print("无 examples 时的结果:", res_no_ex)
# 页面示例显示模型可能会虚构 Person 记录，例如: Data(people=[Person(name='Earth', hair_color='None', height_in_meters='0.00')])
# 使用 deepseek 实际发现，模型并没有出现幻觉，将 Earth 视为人名

# 2) 带上 example messages 后再调用（示例修正行为）
res_with_examples = structured_llm_data.invoke(messages + [message_no_extraction])
print("带 examples 后的结果:", res_with_examples)
# 页面示例显示：Data(people=[])


# 版本 & 模型兼容：确保 langchain-core>=0.3.20 且 provider/模型支持 tool/fn calling。
# LangChain

# Schema 设计：把字段设为 Optional 并写详细 description；必要时用 Enum 或自定义数据类型以减少后处理工作量。
# LangChain

# Few-shot：用 tool_example_to_messages 生成 provider-safe 的示例序列，包含正例和反例以抑制幻觉。
# LangChain

# Tracing & Observability：开启 LangSmith 跟踪，便于追踪每一次 LLM 调用与 tool 调用。
# LangChain

# 鲁棒性：使用低温度、增加验证和后处理（正则/schema 校验/embedding 映射）来提升生产可靠性。
