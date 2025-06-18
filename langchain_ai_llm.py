# pip install --upgrade  openai langchain langchain-openai langchain_community
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()


llm = OpenAI(model="gpt-3.5-turbo")  # 使用补全模型

# 手动将 messages 转为纯文本 prompt
prompt = """你是一个翻译官。请翻译以下内容：
输入: Apple
翻译:

只翻译Apple这一个单词，且只是输出单词的翻译结果
 """
response = llm.invoke(prompt)
print(response)  # 输出: "苹果"
# content='嗨！' response_metadata={'token_usage': {'completion_tokens': 4, 'prompt_tokens': 20, 'total_tokens': 24}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-c714e9bd-465b-4dbb-9441-e0b6e77ebd93-0' usage_metadata={'input_tokens': 20, 'output_tokens': 4, 'total_tokens': 24}
