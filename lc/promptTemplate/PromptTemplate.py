from langchain_core.prompts import PromptTemplate

prompt_template= PromptTemplate.from_template("给我讲一个关于{cotent}的{adjective}的笑话")
result = prompt_template.invoke(input={"cotent": "猴子","adjective":"冷"})
print(result.to_string())
print("..........................")
print(type(result))
                                
