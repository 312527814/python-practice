from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain,ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate.from_template("给做 {product} 的公司起一个名字?")


class MyConstructorCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, prompts, **kwargs):
        print("构造器回调：链开始运行")

    def on_chain_end(self, response, **kwargs):
        print("构造器回调：链结束运行")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("请求回调：模型开始运行")

    def on_llm_end(self, response, **kwargs):
        print("请求回调：模型结束运行")




def constructor_test():
    handler = MyConstructorCallbackHandler()
    # 在构造器中使用回调处理器
    chain = LLMChain(llm=llm, prompt=prompt)
    # 这次运行将使用构造器中定义的回调
    result=chain.run("杯子")
    print(result)



if __name__ == "__main__":
   constructor_test()
