
#pip install llama-index
from llama_index.core import SimpleDirectoryReader

#加载本地文档进行解析
# documents = SimpleDirectoryReader(input_dir = "/root/public/projects/demo_21/data",required_exts=[".txt"]).load_data()

documents = SimpleDirectoryReader(input_dir = "./data").load_data()
#加载某个文档
# documents = SimpleDirectoryReader(input_files="./").load_data()
# print(documents)

for idx, item in enumerate(documents, start=1):
    print(f"{idx}: {item.text}")