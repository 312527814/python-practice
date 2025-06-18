from openai import OpenAI, APIConnectionError, APIError
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=30  # 增加超时时间
)

try:
    response = client.embeddings.create(
        input=["test"],
        model="text-embedding-3-small"
    )
    if response and hasattr(response, 'data'):
        print("前5维嵌入:", response.data[0].embedding[:5])
    else:
        print("错误：响应格式异常", response)

except APIConnectionError as e:
    print(f"连接失败: {e.__cause__}")  # 通常是网络问题
except APIError as e:
    print(f"API 返回错误: {e.status_code} {e.message}")
except Exception as e:
    print(f"未知错误: {type(e).__name__}: {e}")