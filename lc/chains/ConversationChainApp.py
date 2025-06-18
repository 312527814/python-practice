from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# 初始化语言模型
llm = OpenAI(temperature=0.7)

# 创建ConversationChain，默认带有ConversationBufferMemory
conversation = ConversationChain(llm=llm)
print(conversation.invoke({"input":"给做儿童玩具的公司起一个名字"}))
print(conversation.invoke({"input":"给做汽车的公司起一个名字"}))
