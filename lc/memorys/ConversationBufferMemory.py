
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory  # 导入 ConversationBufferMemory 类，用于管理对话缓冲区内存

# 创建一个 ConversationBufferMemory 实例。这里的 `return_messages=True` 表明我们需要返回消息列表以适应 MessagesPlaceholder
# 注意 `"chat_history"` 与 MessagesPlaceholder 的名称对齐。
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 模拟一段对话并保存上下文
memory.save_context({"input": "嗨"}, {"output": "怎么了"})
memory.save_context({"input": "没什么，你呢"}, {"output": "也没什么"})


if __name__ == "__main__":
    messages= memory.load_memory_variables({})
    print(messages)