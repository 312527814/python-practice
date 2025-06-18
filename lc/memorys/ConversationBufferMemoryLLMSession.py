from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain.memory import ConversationBufferMemory  # 导入 ConversationBufferMemory 类，用于管理对话缓冲区内存

# 创建一个 ConversationBufferMemory 实例。这里的 `return_messages=True` 表明我们需要返回消息列表以适应 MessagesPlaceholder
# 注意 `"chat_history"` 与 MessagesPlaceholder 的名称对齐。
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = OpenAI(temperature=0)
# 创建一个字典来存储每个用户的内存实例
user_memory_store = {}
prompt_template = "{input}"

def get_conversation_chain(user_id):
    # 如果该用户还没有内存实例，则创建一个新的
    if user_id not in user_memory_store:
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template),
            memory=memory
        )
        user_memory_store[user_id] = llm_chain
    else:
        # 如果已经存在，直接返回对应的对话链
        llm_chain = user_memory_store[user_id]

    return llm_chain


def handle_user_input(user_id, input_text):
    # 获取对应用户的对话链
    llm_chain = get_conversation_chain(user_id)

    # 进行对话并返回结果
    return llm_chain.invoke({"input": input_text})

print("User 1:", handle_user_input("user1_id", "你好，今天天气怎么样？"))
print("User 1:", handle_user_input("user1_id", "那明天呢？"))


