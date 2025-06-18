from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from typing import Optional
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


# 1. 设置OpenAI模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0
)

# 2. 创建自定义工具函数
def get_current_weather(location: str, unit: Optional[str] = "celsius"):
    """获取指定城市的当前天气（模拟数据）"""

    print("unit:"+unit)
    weather_data = {
        "location": location,
        "temperature": "22",
        "unit": unit,
        "description": "晴朗",
        "humidity": "65%"
    }
    return json.dumps(weather_data, ensure_ascii=False)

def calculate_expression(expression: str):
    """计算数学表达式"""
    try:
        # 安全限制：只允许基本数学运算
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不安全字符"
        
        result = str(eval(expression))
    except Exception as e:
        result = f"计算错误: {str(e)}"
    return result

# 3. 创建工具实例
weather_tool = Tool(
    name="get_current_weather",
    func=get_current_weather,
    description="""当需要查询城市天气时使用此工具。
    输入应为包含location(城市名)和可选unit(单位，celsius或fahrenheit)的JSON字符串。
    例如：{{"location": "北京", "unit": "celsius"}}"""
)

calc_tool = Tool(
    name="calculate",
    func=calculate_expression,
    description="""当需要进行数学计算时使用此工具。
    输入应为数学表达式字符串。
    例如："3 * 5 + 2" """
)

# 创建维基百科工具
# wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# 4. 组合工具列表
# tools = [weather_tool, calc_tool, wikipedia]
tools = [weather_tool, calc_tool]

# 5. 初始化Agent
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.OPENAI_FUNCTIONS,  # 使用OpenAI函数调用风格的Agent
#     verbose=True  # 显示详细执行过程
# )


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 6. 测试对话
queries = [
    "北京现在的天气怎么样？",
    "计算圆周率乘以10的平方是多少？",
    "告诉我爱因斯坦的生平",
    "先查询上海天气，然后计算25华氏度等于多少摄氏度"
]

resutl= agent.run("北京现在的天气怎么样？请用华氏度表示")

print("............1")
print(resutl)
print("...........2")