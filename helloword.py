import json
from lc.promptTemplate.ChatPromptTemplate import my_function2

username = "dsdsds";
print("你的用户名是：" + username)

json_str = '{"id":"chatcmpl-BUTLer38724i6tCRzX6j9mz9C1qsW","object":"chat.completion","created":1746602354,"model": "gpt-3.5-turbo-0125","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I assist you today?"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":14,"completion_tokens":9,"total_tokens":23},"system_fingerprint":"fp_0165350fbb"}'
python_dict = json.loads(json_str)
print(python_dict["choices"][0]["message"]["content"])  # 输出: 李四

my_function2()
