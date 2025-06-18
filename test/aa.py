from langchain_core.prompts import PromptTemplate
template = """以下是一个人类和AI之间的友好对话。AI很健谈，从其上下文中提供了很多具体细节。如果AI不知道答案，它会如实说不知道。如果相关，你会得到有关人类提到的实体的信息。

   相关实体信息:
   {entities}

   对话:
   人类: {input}
   AI:"""
prompt = PromptTemplate(input_variables=["entities", "input"], template=template)

result = prompt.invoke(input={"entities": "猴子","input":"冷"})

print(result)





