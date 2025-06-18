
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()
# print(os.getenv('OPENAI_API_KEY')) 

embeddings = OpenAIEmbeddings()
test_embedding = embeddings.embed_query("test")
print(test_embedding)  