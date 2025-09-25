from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)


completion = client.chat.completions.create(
  model="/root/autodl-tmp/model",
  messages=[
    {"role": "user", "content": "你叫什么名字？"}
  ]
)


print(completion.choices[0].message)