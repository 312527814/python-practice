import os
from langchain_text_splitters.base import TextSplitter
from langchain_community.document_loaders   import TextLoader

from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接文件路径
file_path = os.path.join(current_dir, "text.txt")
print("file_path:"+file_path)

loader= TextLoader(
    file_path=file_path,
    encoding="utf-8"
)
doc= loader.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=8,
#     chunk_overlap=1,
# )

# splitter= CharacterTextSplitter(10,5)
# docs= splitter.split_documents(doc)
# for d in docs:
#     print(".............")
#     print(d.page_content)


# splitter = CharacterTextSplitter(separator=" ")


# 配置CharacterTextSplitter
splitter = CharacterTextSplitter(
    separator=" ",  # 使用两个换行符作为分隔符
    chunk_size=10,   # 每个块的最大字符数
    chunk_overlap=2, # 块之间的重叠字符数
    length_function=len, # 使用len函数来计算字符数
    is_separator_regex=False, # 分隔符不是正则表达式
)

docs =  splitter.split_documents(doc)

for d in docs:
    print(".............")
    print(d)