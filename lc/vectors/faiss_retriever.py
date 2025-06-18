# pip install -U langchain-community faiss-cpu langchain-openai tiktoken
#pip install faiss-cpu
# 如果您需要使用没有 AVX2 优化的 FAISS 进行初始化，请取消下面一行的注释
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

import os



from dotenv import load_dotenv
load_dotenv()

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接文件路径
file_path = os.path.join(current_dir, "qa.txt")


loader = TextLoader(file_path, encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)
query = "Pixar公司是做什么的?"
retriever = db.as_retriever()
docs = retriever.invoke(query)
print(docs[0].page_content)