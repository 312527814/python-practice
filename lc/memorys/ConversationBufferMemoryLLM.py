from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain.memory import ConversationBufferMemory  # 导入 ConversationBufferMemory 类，用于管理对话缓冲区内存

# 创建一个 ConversationBufferMemory 实例。这里的 `return_messages=True` 表明我们需要返回消息列表以适应 MessagesPlaceholder
# 注意 `"chat_history"` 与 MessagesPlaceholder 的名称对齐。
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

print(memory.load_memory_variables({}))
print("......................1")
prompt_template = "{input}"

llm = OpenAI(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
    memory=memory
    )
print(llm_chain.invoke({"input":"给做儿童玩具的公司起一个名字?"}))
print("......................2")
print(memory.load_memory_variables({}))

print(llm_chain.invoke({"input":"从你给的这些名字中选一个吧"}))

print("......................3")
print(memory.load_memory_variables({}))


