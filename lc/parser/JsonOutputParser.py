from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


from dotenv import load_dotenv
load_dotenv()

# 定义 Pydantic 模型
class Answer(BaseModel):
    definition: str = Field(description="术语的定义")
    example: str = Field(description="术语的示例")

# 初始化 LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "将回答格式化为 JSON，包含定义和示例。"),
    ("human", "解释人工智能：{question}")
])

# 创建 LCEL 链
parser = JsonOutputParser(pydantic_object=Answer)
chain = prompt | llm | parser

# 调用链
response = chain.invoke({"question": "什么是人工智能？"})
print(response)



