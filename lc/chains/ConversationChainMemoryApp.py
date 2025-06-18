from langchain.chains import ConversationChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory  # 导入 ConversationBufferMemory 类，用于管理对话缓冲区内存

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# 初始化语言模型
llm = OpenAI(temperature=0.7)



# 创建一个 ConversationBufferMemory 实例。这里的 `return_messages=True` 表明我们需要返回消息列表以适应 MessagesPlaceholder
# 注意 `"chat_history"` 与 MessagesPlaceholder 的名称对齐。
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt_template = "{input}"
# 创建ConversationChain，默认带有ConversationBufferMemory
conversation = ConversationChain(llm=llm)

print(conversation.invoke({"input":"你好，今天天气怎么样？"}))
print("................1")
print(conversation.invoke(input= "那明天呢？"))

