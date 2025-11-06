from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime  
from langchain.agents import create_agent
@dataclass
class Context:
    user_id: str

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:  
    """Fetch the user's email preferences from the store."""
    user_id = runtime.context.user_id  

    preferences: str = "这位用户希望你写一封简短而礼貌的电子邮件。"
    if runtime.store:  
        if memory := runtime.store.get(("users",), user_id):  
            preferences = memory.value["preferences"]

    return preferences

agent = create_agent(
    model="gpt-5-nano",
    tools=[fetch_user_email_preferences],
    context_schema=Context
)

# 模拟运行
result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我写封邮件"}]},
    context=Context(user_id="u1234")
)

for message in result["messages"]:
    message.pretty_print()