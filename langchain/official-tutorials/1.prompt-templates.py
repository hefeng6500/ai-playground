from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

# print(prompt_template.invoke({"topic": "cats"}))

# ChatPromptTemplates

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant"),
        ("user", "Tell me a joke about {topic}"),
    ]
)

# print(prompt_template.invoke({"topic": "cats"}))


# MessagesPlaceholder

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt_template = ChatPromptTemplate(
    [("system", "You are a helpful assistant"), MessagesPlaceholder("msgs")]
)

# MessagesPlaceholder("msgs")：它不是一条具体的消息，而是一个 占位符。运行时它会被替换成你传进去的消息列表（HumanMessage, AIMessage, 甚至 SystemMessage）。所以它自己不能指定一个固定角色，否则就无法容纳不同角色的历史消息。

# Simple example with one message
prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})

# More complex example with conversation history
messages_to_pass = [
    HumanMessage(content="What's the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="And what about Germany?"),
]

formatted_prompt = prompt_template.invoke({"msgs": messages_to_pass})
print(formatted_prompt)
