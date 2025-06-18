from langchain.chains import SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.globals import set_verbose
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
llm = ChatOpenAI(temperature=0.9)

#创建两个子链
# 提示模板 1 ：这个提示将接受产品并返回最佳名称来描述该公司
first_prompt = ChatPromptTemplate.from_template(
    "描述制造{product}的一个{unit}的最好的名称是什么"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt,output_key="company")

# 提示模板 2 ：接受公司名称，然后输出该公司的长为20个单词的描述
second_prompt = ChatPromptTemplate.from_template(
    "写一个20字的描述对于下面这个\
    公司：{company}的"
)
# 打印详细日志
set_verbose(True)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

#构建简单顺序链，组合两个LLMChain，以便我们可以在一个步骤中创建公司名称和描述
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True)

product = "大号床单套装"
result= chain_one.invoke({"product":product,"unit":"公司"})
print("........................")
print(result)