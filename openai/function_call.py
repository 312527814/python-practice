from openai import OpenAI
import json
import math
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# 1. 定义可用的函数（工具）
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """获取指定城市的当前天气（模拟数据）"""

    weather_data = {
        "location": location,
        "temperature": "22",
        "unit": unit,
        "description": "晴朗",
        "humidity": "65%"
    }
    return json.dumps(weather_data)

def calculate_expression(expression: str):
    """计算数学表达式"""
    try:
        result = str(eval(expression))
    except Exception as e:
        result = f"计算错误: {str(e)}"
    return json.dumps({"expression": expression, "result": result})

# 2. 定义函数规范
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，摄氏度或华氏度",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：3 + 5 * 2",
                    },
                },
                "required": ["expression"],
            },
        },
    }
]

# 3. 主对话函数
def run_conversation(prompt):
    messages = [{"role": "user", "content": prompt}]
    
    # 第一步：发送给模型，让模型决定是否调用函数
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # 如果有函数调用
    if tool_calls:
        messages.append(response_message)
        
        # 处理每个函数调用
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(function_args)
            
            # 调用对应的函数
            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit", "celsius")
                )
            elif function_name == "calculate_expression":
                function_response = calculate_expression(
                    expression=function_args.get("expression")
                )
            
            # 将函数响应添加到消息中
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        
        print("messages..1....................")

        print(messages)
        print("messages......2................")

        # 将函数响应发送回模型获取最终回答
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        
        return second_response.choices[0].message.content
    
    return response_message.content

# 4. 测试对话
queries = [
    "北京现在的天气怎么样？",
    "计算一下3的平方加上4的平方等于多少？",
    "先告诉我上海天气，然后计算25度华氏度等于多少摄氏度？"
]

# for query in queries:
#     print(f"用户: {query}")
#     response = run_conversation(query)
#     print(f"AI助手: {response}\n")

response = run_conversation("北京现在的天气怎么样？")

print(response)