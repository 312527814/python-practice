from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from typing import Optional
import json
import math
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()


# 1. 设置OpenAI模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0
)

# 2. 创建自定义工具函数
def deleteOrder(orderId: str):
    """删除订单"""
    print(orderId+":订单删除成功")
    return "订单删除成功"


def addOrder(input_dict: dict):
    """添加订单（接受字典参数）"""
    order= json.loads(input_dict)

    orderName = order["orderName"]
    user = order["user"]
    return user+"的"+orderName+"订单创建成功。请以优美的句子回应他，显的我们很重视"

# 3. 创建工具实例
delete_order_tool = Tool(
    name="deleteOrder",
    func=deleteOrder,
    description="""当需要查询订单的时候调用此工具。
    输入应为包含orderId(订单id)字符串。
    """
)

add_order_tool = Tool(
    name="addOrder",
    func=addOrder,
    description="""当需要添加订单的时候调用此工具。
    输入应为包含orderName(订单名称)和user(订单人)的JSON字符串。
    例如：{{"orderName": "苹果", "user": "张三"}}"""
)



# 4. 组合工具列表
# tools = [weather_tool, calc_tool, wikipedia]
tools = [delete_order_tool,add_order_tool]

# 5. 初始化Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True  # 显示详细执行过程
)

# 6. 测试对话
# resutl= agent.run("删除订单1")
resutl= agent.invoke("给李四添加一个香蕉订单")
print("............1")
print(resutl)
print("...........2")