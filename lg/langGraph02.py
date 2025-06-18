from typing import Literal
from langchain_core.messages import HumanMessage,ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# pip install langgraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


from dotenv import load_dotenv
load_dotenv()
a=0
# 定义工具函数，用于代理调用外部工具
# def search(state:MessagesState):
#     """模拟一个搜索工具"""
#     global a
#     messages= state["messages"]
#     if a==0:
#         toolMessage = HumanMessage(content='现在30度，有雾.')
#     else:
#         toolMessage= HumanMessage(content='现在30度，有雨.')
#     messages.append(toolMessage)
#     a=a+1
#     print("...............12state")
#     print(state)
#     print("...............22state")
#     return state

def search(state: MessagesState):
    messages = state["messages"]
    global a

    if a==0:
      tool_message = HumanMessage(content='现在30度，有雾.')
    else:
      tool_message= HumanMessage(content='现在30度，有雨.')
    messages.append(tool_message)
    a=a+1
    return {"messages": messages}

# 1.初始化模型和工具，定义并绑定工具到模型
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# 定义调用模型的函数
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # 返回列表，因为这将被添加到现有列表中
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["agent", END]:
    messages = state['messages']
    last_message = messages[-1]
    if "有雾" in last_message.content:
        return "agent"
    # 否则，停止（回复用户）
    return END

# 2.用状态初始化图，定义一个新的状态图
workflow = StateGraph(MessagesState)


# 3.定义图节点，定义我们将循环的两个节点
workflow.add_node("agent", call_model)
workflow.add_node("search_weather_node", search)

# 4.定义入口点和图边
# 设置入口点为“agent”
# 这意味着这是第一个被调用的节点
workflow.set_entry_point("agent")

# 添加从`tools`到`agent`的普通边。
# 这意味着在调用`tools`后，接下来调用`agent`节点。
workflow.add_edge("agent", 'search_weather_node')

# 添加条件边
workflow.add_conditional_edges(
    # 首先，定义起始节点。我们使用`agent`。
    # 这意味着这些边是在调用`agent`节点后采取的。
    "search_weather_node",
    # 接下来，传递决定下一个调用节点的函数。
    should_continue,
)



# 初始化内存以在图运行之间持久化状态
checkpointer = MemorySaver()

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

graph_png = app.get_graph().draw_mermaid_png()
with open("node_case.png", "wb") as f:
    f.write(graph_png)
