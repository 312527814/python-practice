from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

# 分阶段处理：GPT-4 生成 → Claude 优化
chain = (
    ChatPromptTemplate.from_template("生成关于{topic}的10个要点") 
    | model 
    | ChatPromptTemplate.from_template("用更简洁的语言优化以下内容:\n\n{input}")
    | model
)

result = chain.invoke({"topic": "量子计算"})
print(result.content)