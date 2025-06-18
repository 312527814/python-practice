from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
#字典作为输入并返回消息作为输出
runnable = prompt | model
# print(runnable)
history=[HumanMessage(content="我是张三")]


result = runnable.invoke({"history":history ,"ability":"talk","input":"请问我是谁？"})
history.append(result)


result = runnable.invoke({"history":history ,"ability":"talk","input":"我喜欢李四，我应该怎么讨好他？"})

history.append(HumanMessage(content="我喜欢李四，我应该怎么讨好他？"))
history.append(result)


print(history)
result = runnable.invoke({"history":history ,"ability":"talk","input":"请问我喜欢谁？"})

print(result.content)

