from typing import TypedDict
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

class State(TypedDict):
   """The graph state."""
   some_text: str

def human_node(state: State):
   value = interrupt(
      # Any JSON serializable value to surface to the human.
      # For example, a question or a piece of text or a set of keys in the state
      {
         "text_to_revise": state["some_text"]
      }
   )
   return {
      # Update the state with the human's input
      "some_text": value+"ddd"
   }


# Build the graph
graph_builder = StateGraph(State)
# Add the human-node to the graph
graph_builder.add_node("human_node", human_node)
graph_builder.add_edge(START, "human_node")

# A checkpointer is required for `interrupt` to work.
checkpointer = MemorySaver()
graph = graph_builder.compile(
   checkpointer=checkpointer
)

# Pass a thread ID to the graph to run it.
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

# Using stream() to directly surface the `__interrupt__` information.
for chunk in graph.stream({"some_text": "Original text"}, config=thread_config):
   print("是否有中断？"+str(chunk["__interrupt__"] is not None ))# 如果__interrupt__ 这个key说明有中断"
   print("中断返回的值："+str(chunk["__interrupt__"][0].value))# 中断返回的内容


print(".........................")

# Resume using Command
for chunk in graph.stream(Command(resume="Edited text"), config=thread_config):
   print(chunk)

