from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# 初始化语言模型实例
llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")

# 创建自定义的 PromptTemplate
custom_prompt = PromptTemplate(
    input_variables=["history", "input"],  # 输入变量包括对话历史和当前输入
    template="以下是我们的对话历史和当前的问题。\n"
             "对话历史:\n{history}\n"
             "问题: {input}\n"
             "请根据上述信息回答问题:"
)
# custom_prompt=PromptTemplate.from_template("啊啊{input},{history}")
# 创建内存实例
memory = ConversationBufferMemory()

# 创建带有自定义提示的 ConversationChain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=custom_prompt,
    verbose=True  # 开启详细日志，方便调试
)

# 进行对话
response = conversation.run(input="你好，今天天气怎么样？")
print(response)
response2 = conversation.run(input="那明天呢？")
print(response2)

response3 = conversation.invoke(input="那后天呢？")
print(response3)