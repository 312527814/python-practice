from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
prompt = PromptTemplate(
    input_variables=["question"],
    template="回答以下问题：{question}"
)
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)
result = llm_chain.run(question="法国的首都是哪里？")
print(result)

print("............1")
result2 = llm_chain.predict(question="法国的首都是哪里?")
print(result2)

print("............2")
result3 = llm_chain.invoke({"question": "法国的首都是哪里？"})

print(result3)
