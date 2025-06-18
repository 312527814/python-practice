from typing import Literal
from langchain_core.messages import HumanMessage,ToolMessage,BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# pip install langgraph
from langgraph.checkpoint.memory import MemorySaver

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Dict, Any
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

pool = ConnectionPool(conninfo="postgresql://postgres:3tXSJTUs@10.116.17.84:5432/postgres", max_size=20)
# checkpointer = PostgresSaver(sync_connection=pool)

checkpointer = PostgresSaver.from_conn_string("postgresql://postgres:3tXSJTUs@10.116.17.84:5432/postgres")

checkpointer.setup()

from dotenv import load_dotenv
load_dotenv()


class MyState(TypedDict):
    messages: list[BaseMessage]

def search(state: MyState):
    messages = state["messages"]
    tool_message = HumanMessage(content='现在30度，有雾.')
    messages.append(tool_message)
    return state

def summary(state: MyState):
    messages = state["messages"]
    tool_message = HumanMessage(content='我问的是什么问题？')
    messages.append(tool_message)
    return state

# 1.初始化模型和工具，定义并绑定工具到模型
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# 定义调用模型的函数
def call_model(state: MyState):
    messages = state['messages']
    response = model.invoke(messages)
    messages.append(response)
    # 返回列表，因为这将被添加到现有列表中
    return state

# 2.用状态初始化图，定义一个新的状态图
workflow = StateGraph(MyState)


# 3.定义图节点，定义我们将循环的两个节点
workflow.add_node("agent1", call_model)
workflow.add_node("agent2", call_model)
workflow.add_node("search_weather_node", search)
workflow.add_node("summary", summary)

# 4.定义入口点和图边
# 设置入口点为“agent”
# 这意味着这是第一个被调用的节点
workflow.set_entry_point("agent1")

# 添加从`tools`到`agent`的普通边。
# 这意味着在调用`tools`后，接下来调用`agent`节点。
workflow.add_edge("agent1", 'search_weather_node')


workflow.add_edge("search_weather_node", 'summary')
workflow.add_edge("summary", 'agent2')


# 初始化内存以在图运行之间持久化状态
# checkpointer = MemorySaver()

# 5.编译图
# 这将其编译成一个LangChain可运行对象，
# 这意味着你可以像使用其他可运行对象一样使用它。
# 注意，我们（可选地）在编译图时传递内存
app = workflow.compile(checkpointer=checkpointer)

# 6.执行图，使用可运行对象
final_state = app.invoke(
    {"messages": [HumanMessage(content="上海的天气怎么样?")]},
    config={"configurable": {"thread_id": 42}}
)

# 从 final_state 中获取最后一条消息的内容
result = final_state["messages"][-1].content
print(final_state)
print(type(final_state))

graph_png = app.get_graph().draw_mermaid_png()
with open("node_case_03.png", "wb") as f:
    f.write(graph_png)
