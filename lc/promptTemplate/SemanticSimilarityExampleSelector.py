

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

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
examples_selector = SemanticSimilarityExampleSelector.from_examples(
    # 这是可供选择的示例列表
    examples,
    # 这是用于生成嵌入的嵌入类，该嵌入用于衡量语义相似性
    OpenAIEmbeddings(),
    # 这是用来存储嵌入和执行相似性搜索的VectorStore类
    Chroma,
    # 这是要生成的示例数
    k=1
)
question = "乔治·华盛顿的父亲是谁"
example_prompt = PromptTemplate.from_template("内容：{question}\n{answer}")


prompt = FewShotPromptTemplate(
    example_selector=examples_selector,
    example_prompt=example_prompt,
    suffix="问题：{input}"
)
result= prompt.format(input=question)

print(result)

