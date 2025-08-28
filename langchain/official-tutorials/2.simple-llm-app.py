from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


model = init_chat_model("deepseek-chat", model_provider="deepseek")

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("Hello World!"),
]

# model.invoke(messages)

# Stream 输出
# for token in model.stream(messages):
#     print(token.content, end="|")

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

# 此提示模板的输入是一个字典
print(prompt)

# 返回了一个由两个消息组成的 ChatPromptValue
print(prompt.to_messages())

response = model.invoke(prompt)
print(response.content)
