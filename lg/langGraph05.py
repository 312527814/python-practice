from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str

def step_1(state):
    print("---Step 1---")
    pass

def step_2(state):
    print("---Step 2---")
    # return {"input":"哈哈哈"}
    pass

def step_3(state):
    print("---Step 3---")
    pass

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add
graph = builder.compile(checkpointer=memory, interrupt_before=["step_3"])

# Input
initial_input = {"input": "hello world"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# 运行graph，直到第一次中断
# for event in graph.stream(initial_input, thread, stream_mode="values"):
#     print(event)
# 6.执行图，使用可运行对象

final_state = graph.invoke(
    initial_input,
    config=thread
)
print(final_state)
print(type(final_state))
user_approval = input("Do you want to go to Step 3? (yes/no): ")

if user_approval.lower() == "yes":
    final_state = graph.invoke(
        initial_input,
        config=thread
    )
else:
    print("Operation cancelled by user.")
