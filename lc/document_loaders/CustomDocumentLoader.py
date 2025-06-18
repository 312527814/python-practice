from typing import AsyncIterator, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import asyncio


class CustomDocumentLoader(BaseLoader):
    """一个从文件逐行读取的示例文档加载器。"""

    def __init__(self, file_path: str) -> None:
        """使用文件路径初始化加载器。
        Args:
            file_path: 要加载的文件的路径。
        """
        self.file_path = file_path

    def load(self) -> list[Document]:  # <-- 不接受任何参数
        """逐行读取文件的惰性加载器。
        当您实现惰性加载方法时，应使用生成器逐个生成文档。
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                # yield 关键字用于定义生成器函数。
                # 生成器函数是一种特殊类型的函数，它允许你逐步生成一个序列的值，而不是一次性返回整个序列。
                # 与普通的函数不同，生成器函数在每次调用时不会从头开始执行，而是从上次离开的地方继续执行。
                # 这使得生成器非常适合处理需要逐步生成或处理大数据集的情况

                print("1............... "+str(line_number))
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                print("2............... " + str(line_number))
                line_number += 1


loader = CustomDocumentLoader("./customLoaderText.txt")

docs=loader.load();

## 测试延迟加载接口
for doc in docs:
    print(doc.page_content)
    print(type(doc))
    print(doc)
    print("................")

