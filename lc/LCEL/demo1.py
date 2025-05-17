from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),
      ("human", "{text}")
    ]
)

# 第二段：翻译文本
translate_prompt =ChatPromptTemplate.from_template("用更简洁的语言优化以下内容:\n\n{input}")

chain=prompt_template|model|translate_prompt|model
result = chain.invoke({"language": "Chinese", "text": "hi"})
print(result.content)