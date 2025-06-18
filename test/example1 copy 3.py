# 导入langchain库中的相关模块
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain.tools.render import format_tool_to_openai_function
from langchain.schema.agent import AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from dotenv import load_dotenv 

# 加载环境变量
load_dotenv()

# 定义一个工具函数，用于获取句子中不同汉字的数量
@tool
def count_unique_chinese_characters(sentence)-> int:
    """用于计算句子中不同汉字的数量"""
    unique_characters = set()

    # 遍历句子中的每个字符
    for char in sentence:
        # 检查字符是否是汉字
        if '\u4e00' <= char <= '\u9fff':
            unique_characters.add(char)

    # 返回不同汉字的数量
    return len(unique_characters)



# 创建一个聊天提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_output"),
    ]
)

# 构建一个代理，它将处理输入、提示、模型和输出解析
agent = (
    {
        "input": lambda x: x["input"],
        "agent_output": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
)
sentence = "‘如何用LangChain实现一个代理’计算这句话共包含几个不同的汉字"

from langchain_core.agents import AgentAction, AgentActionMessageLog

intermediate_steps=[ (AgentAction("tool",tool_input="tool_input", log="log"), "好的")]
result= agent.invoke({"input":sentence,"intermediate_steps":intermediate_steps})

print(result)
print(type(result)) #PromptValue




