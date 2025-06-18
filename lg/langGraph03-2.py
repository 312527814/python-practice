from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage
from psycopg_pool import ConnectionPool

# 定义状态
class State(TypedDict):
    messages: List[dict]

# 定义节点
def agent_node(state: State) -> State:
    last_message = state["messages"][-1]["content"]
    return {"messages": state["messages"] + [{"role": "assistant", "content": "Echo: " + last_message}]}

# 创建连接池
pool = ConnectionPool(conninfo="postgresql://postgres:3tXSJTUs@10.116.17.84:5432/postgres", max_size=20)

# 初始化 PostgresSaver（langgraph >= 0.2）
checkpointer = PostgresSaver(sync_connection=pool)
checkpointer.setup()  # 自动创建所需表结构（如果不存在）

# 构建状态图
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.set_finish_point("agent")

# 编译图
graph = builder.compile(checkpointer=checkpointer)

# 运行多轮对话
config = {"configurable": {"thread_id": "thread-1"}}
result1 = graph.invoke({"messages": [{"role": "user", "content": "Hello"}]}, config=config)
print(result1["messages"][-1]["content"])  # 输出: Echo: Hello

result2 = graph.invoke({"messages": [{"role": "user", "content": "How are you?"}]}, config=config)
print(result2["messages"][-1]["content"])  # 输出: Echo: How are you?