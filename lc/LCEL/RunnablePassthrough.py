from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

def function1():
    runablePass= RunnablePassthrough()

    runnable =runablePass

    result= runnable.invoke({"cotent": "猴子","adjective":"冷"})

    print(result)

def function2():
    chain = RunnableParallel(
    original_input=RunnablePassthrough(),  # 保留原始输入
    processed=lambda x: x.upper(),        # 同步处理输入
    length=lambda x: len(x)               # 计算输入长度
    )
    result= chain.invoke("hello")
    print(result)

function1()
function2()
