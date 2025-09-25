from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import  os
from dotenv import load_dotenv
load_dotenv()

api_key= os.getenv("ModelScope_API_KEY")
base_url="https://api-inference.modelscope.cn/v1/"

model = ChatOpenAI(model="Qwen/Qwen2.5-Coder-32B-Instruct",api_key=api_key,base_url=base_url)

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),
      ("human", "{text}")
    ]
)

# 第二段：翻译文本
translate_prompt =ChatPromptTemplate.from_template("用更简洁的语言优化以下内容:\n\n{input}")

chain=prompt_template|model|translate_prompt|model
result = chain.invoke({"language": "Chinese", "text": "we should do everything we can to saved our city"})
print(result.content)