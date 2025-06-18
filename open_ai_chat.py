from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用的模型名称
        messages=[
            {"role": "system", "content": "assistant"},  # 系统消息
            {"role": "user", "content": "Hello world"}  # 用户输入的提示
        ],
        max_tokens=150  # 最大生成的标记数
    )
    # 返回生成的响应文本
print(response.choices[0].message.content.strip())