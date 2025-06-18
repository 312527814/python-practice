from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

response = client.completions.create(
        model="gpt-3.5-turbo",  # 使用的模型名称
        prompt="请告诉我你是谁？",
        max_tokens=150  # 最大生成的标记数
    )
    # 返回生成的响应文本
print(response)