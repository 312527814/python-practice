from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# 定义一个函数来获取外部数据
def fetch_data(query):
    # response = requests.get(f"https://api.example.com/data?query={query}")
    # return response.json()
    return "dsdsd"


# 第一步：根据用户输入生成查询
first_prompt = PromptTemplate(
    input_variables=["input"],
    template="将用户的提问转换为API查询: {input}"
)

# 初始化语言模型
llm = OpenAI(temperature=0.7)

# 创建第一个链（生成查询）
first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="data")

# 第二步：使用查询获取数据并生成最终答案
second_prompt = PromptTemplate(
    input_variables=["query"],
    template="查询结果是: {data}\n基于上述查询结果，请回答以下问题: {query}"
)

# 创建第二个链（获取数据并生成答案）
second_chain = LLMChain(
    llm=llm,  # 注意这里应为llm而不是lll
    prompt=second_prompt,
    output_key="final_answer"
)

# 创建一个顺序链来组合这两个步骤
overall_chain = SequentialChain(
    chains=[first_chain, second_chain],
    input_variables=["input"],
    output_variables=["final_answer"],
    verbose=True
)


# 定义一个函数来执行整个流程
def process_user_input(user_input):
    # 运行整个链条
    return overall_chain({"input": user_input})

    # 获取数据
    # query = result["query"]
    # data = fetch_data(query)
    #
    # # 将数据作为额外的输入传递给第二个链
    # final_response = second_chain.run({"query": user_input, "data": str(data)})
    # return final_response


# 测试
response = process_user_input("今天的天气怎么样？")
print(response)