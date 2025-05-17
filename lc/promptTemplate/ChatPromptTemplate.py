from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate





def my_function():
    system_template = "Translate the following into {language}:"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template),
          ("human", "{text}"),
          ("ai", "{text}")]
    )
    result = prompt_template.invoke({"language": "Chinese", "text": "hi"})
    print(result)



def my_function2():
    system_template = "Translate the following into {language}:"

    msg =  [("system", system_template),("human", "{text}"), ("ai", "{text}")]

    prompt_template = ChatPromptTemplate.from_messages(
       msg
    )
    result = prompt_template.invoke({"language": "Chinese", "text": "hi"})
    print(result)

def my_function3():
    prompt_template = ChatPromptTemplate.from_messages(
        [
        SystemMessage(
            content=(
                "你是一个乐于助人的助手，可以润色内容，使其看起来更简单易读"
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    )
    result = prompt_template.invoke({"text":"我不喜欢吃难吃的东西"})
    print(result)
   

# my_function()
# my_function2()
my_function3()


