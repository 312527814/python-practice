from langchain_core.runnables import RunnableParallel, RunnablePassthrough

chain = RunnableParallel(
    original_input=RunnablePassthrough(),  # 保留原始输入
    processed=lambda x: print(x),        # 同步处理输入
    length=lambda x: len(x)               # 计算输入长度
)
# chain = RunnableParallel(
#     processed=lambda x: x.upper(),        # 同步处理输入
#     length=lambda x: len(x)               # 计算输入长度
# )

result= chain.invoke({"text": "hello"})

print(result)