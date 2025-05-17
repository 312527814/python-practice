from langchain_core.messages import HumanMessage
from langchain.prompts import HumanMessagePromptTemplate

def create1():
    message=HumanMessage(content="我是humanMessage")
    print(message)
    print(type(message))

def create2():
     message_template=HumanMessagePromptTemplate.from_template("我是humanMessage")
     message=message_template.format()
     print(message)
     print(type(message))

def create3():
    message=HumanMessage(content="我是{text}")
    print(message)
    print(type(message))

def create4():
     message_template=HumanMessagePromptTemplate.from_template("我是{text}")
     message=message_template.format(text="humanMessage")
     print(message)
     print(type(message))

create4()
#create2()                                    
