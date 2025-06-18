
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()




# 初始化 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")  # 使用补全模型

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，请回答问题。"),
    ("human", "{question}")
])

# 创建 LCEL 链
chain = prompt | llm | StrOutputParser()

agent = ( prompt
    | llm
    | StrOutputParser()
)
agent.invoke

# chain = prompt | llm

# 调用链
response = chain.invoke({"question": "什么是人工智能？返回不要超过10个字"})
print(response)
