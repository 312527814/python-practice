from langchain import hub as prompts
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
prompt = prompts.pull("my1")
model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model
print(chain.invoke({"question": "你是谁？"}))