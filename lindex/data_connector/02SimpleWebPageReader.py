# pip install llama-index
# 加载网页数据
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=True).load_data(
        ["https://finance.eastmoney.com/a/202502033310108421.html"]
    )

for idx, item in enumerate(documents, start=1):
        print(f"{idx}: {item.get_content()}")