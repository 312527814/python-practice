import os
from langchain_community.document_loaders   import TextLoader

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接文件路径
file_path = os.path.join(current_dir, "text.txt")
loader= TextLoader(
    file_path=file_path,
    encoding="utf-8"
)
data= loader.load()

print(data)

# for record in data:
#     print(record.page_content)


