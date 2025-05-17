from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel
from langchain.output_parsers  import CommaSeparatedListOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt_tpl_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出{cityName}的{viewPointNum}个著名景点。"),
    ]
)
prompt_tpl_2 = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出关于{cityName}历史的{viewPointNum}个著名书籍。"),
    ]
)

output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()

model = ChatOpenAI(model="gpt-3.5-turbo")

chain_1 = prompt_tpl_1 | model | output_parser
chain_2 = prompt_tpl_2 | model | output_parser
chain_parallel = RunnableParallel(view_point=chain_1, book=chain_2)

response = chain_parallel.invoke(
    {"cityName": "南京", "viewPointNum": 3, "parser_instructions": parser_instructions}
)

print(response)