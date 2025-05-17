

examples = [
{
"question": "谁活得更⻓，穆罕默德·阿⾥还是艾伦·图灵？",
"answer": """
是否需要后续问题：是的。
后续问题：穆罕默德·阿⾥去世时多⼤年纪？
中间答案：穆罕默德·阿⾥去世时74岁。
后续问题：艾伦·图灵去世时多⼤年纪？
中间答案：艾伦·图灵去世时41岁。
所以最终答案是：穆罕默德·阿⾥
""",
},
{
"question": "克雷格斯列表的创始⼈是什么时候出⽣的？",
"answer": """
是否需要后续问题：是的。
后续问题：克雷格斯列表的创始⼈是谁？
中间答案：克雷格斯列表的创始⼈是克雷格·纽⻢克。
后续问题：克雷格·纽⻢克是什么时候出⽣的？
中间答案：克雷格·纽⻢克于1952年12⽉6⽇出⽣。
所以最终答案是：1952年12⽉6⽇
""",
},
{
"question": "乔治·华盛顿的外祖⽗是谁？",
"answer": """
是否需要后续问题：是的。
后续问题：乔治·华盛顿的⺟亲是谁？
中间答案：乔治·华盛顿的⺟亲是玛丽·波尔·华盛顿。
后续问题：玛丽·波尔·华盛顿的⽗亲是谁？
中间答案：玛丽·波尔·华盛顿的⽗亲是约瑟夫·波尔。
所以最终答案是：约瑟夫·波尔
""",
},
{
"question": "《⼤⽩鲨》和《皇家赌场》的导演都来⾃同⼀个国家吗？",
"answer": """
是否需要后续问题：是的。
后续问题：《⼤⽩鲨》的导演是谁？
中间答案：《⼤⽩鲨》的导演是史蒂⽂·斯⽪尔伯格。
后续问题：史蒂⽂·斯⽪尔伯格来⾃哪个国家？
中间答案：美国。
后续问题：《皇家赌场》的导演是谁？
中间答案：《皇家赌场》的导演是⻢丁·坎⻉尔。
后续问题：⻢丁·坎⻉尔来⾃哪个国家？
中间答案：新⻄兰。
所以最终答案是：不是
""",
},
]

from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
example_prompt = PromptTemplate.from_template("内容：{question}\n{answer}")
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="问题：{input}",
)
result= prompt.invoke({"input": "乔治·华盛顿的⽗亲是谁？"})
print(result)
print(type(result))
#print(prompt.invoke({"input": "乔治·华盛顿的⽗亲是谁？"}).to_string())

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model="gpt-3.5-turbo")

chain=prompt|model
response = chain.invoke({"input": "乔治·华盛顿的⽗亲是谁？直接告诉我具体名字"})
print(response.content)