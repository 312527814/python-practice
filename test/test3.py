from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
prompt = PromptTemplate(
    template="回答以下问题：{question}"
)
PromptTemplate.from_template("")
llm=OpenAI()
llm_chain = prompt|llm
result = llm_chain.invoke({"question":"法国的首都是哪里？"})
print(result)