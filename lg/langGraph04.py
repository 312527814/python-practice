from typing import Literal
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_openai import ChatOpenAI
# pip install langgraph
from langgraph.graph import END, StateGraph
from typing import TypedDict

import redis
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
import  pickle

import redis
import msgpack
from typing import Optional, Dict, Any, Tuple, List
from langgraph.types import   StateSnapshot
from langgraph.checkpoint.base import BaseCheckpointSaver, ChannelVersions
from uuid import UUID
from datetime import datetime
from langchain_core.messages import HumanMessage,BaseMessage

from dotenv import load_dotenv
load_dotenv()


from langgraph.checkpoint.base import CheckpointTuple

class RedisSaver(BaseCheckpointSaver):
    def __init__(self, redis_client):
        self.redis = redis_client

    def save(self, checkpoint: Checkpoint) -> None:
        print("save.............")
        self.redis.set(f"checkpoint:{checkpoint['id']}", pickle.dumps(checkpoint))

    def load(self, checkpoint_id: str) -> Checkpoint:
        print("load.............")
        data = self.redis.get(f"checkpoint:{checkpoint_id}")
        return pickle.loads(data) if data else None

    def get_tuple(self, config: dict) -> Optional[Tuple[dict, StateSnapshot]]:
        thread_id = config["configurable"]["thread_id"]
        key = f"checkpoint:{thread_id}"

        data = self.redis.get(key)
        if not data:
            return None

        checkpoint_data = msgpack.unpackb(data, raw=False)
        latest_checkpoint = checkpoint_data.get("latest", None)

        if not latest_checkpoint:
            return None

        # 构造 StateSnapshot 对象
        state_snapshot = StateSnapshot(
            values=latest_checkpoint["values"],
            next=latest_checkpoint["next"],
            config={"configurable": {"thread_id": thread_id}},
            metadata=latest_checkpoint.get("metadata", {}),
            parent_config=latest_checkpoint.get("parent_config", None),
            versions=ChannelVersions(latest_checkpoint.get("versions", {})),
        )

        return (
            config,
            state_snapshot,
        )
class RedisCheckpointSaver(BaseCheckpointSaver):
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)
        self.ttl = ttl  # seconds to keep checkpoints in Redis

    def get_tuple(self, config: dict) -> Optional[Tuple[dict, StateSnapshot]]:
        thread_id = config["configurable"]["thread_id"]
        key = f"checkpoint:{thread_id}"

        data = self.redis_client.get(key)
        if not data:
            return None

        checkpoint_data = msgpack.unpackb(data, raw=False)
        latest_checkpoint = checkpoint_data.get("latest", None)

        if not latest_checkpoint:
            return None

        # 构造 StateSnapshot 对象
        state_snapshot = StateSnapshot(
            values=latest_checkpoint["values"],
            next=latest_checkpoint["next"],
            config={"configurable": {"thread_id": thread_id}},
            metadata=latest_checkpoint.get("metadata", {}),
            parent_config=latest_checkpoint.get("parent_config", None),
            versions=ChannelVersions(latest_checkpoint.get("versions", {})),
        )

        return (
            config,
            state_snapshot,
        )

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
REDIS_URL = "redis://:Gsafety.123@localhost:7903/0"
redis_client = redis.from_url(REDIS_URL)

# 初始化内存以在图运行之间持久化状态
checkpointer = RedisSaver(redis_client= redis_client)

# checkpointer.setup()

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
with open("node_case_04.png", "wb") as f:
    f.write(graph_png)
