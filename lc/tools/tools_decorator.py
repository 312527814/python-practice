from langchain_core.tools import tool
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


@tool
def multiply(a: int) -> int:
    """Multiply two numbers."""
    return a


# 让我们检查与该工具关联的一些属性。
print(multiply.name)
print(multiply.description)
print(multiply.args)




# 1. 设置OpenAI模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0
)


agent = initialize_agent(multiply, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 6. 测试对话
resutl= agent.run("计算2*2等于多少")

print("............1")
print(resutl)
print("...........2")
