class MyCollection:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return MyIterator(self.items)


class MyIterator:
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.items):
            result = self.items[self.index]
            self.index += 1
            return result
        raise StopIteration


# 使用示例
collection = MyCollection(['apple', 'banana', 'cherry'])
for idx, item in enumerate(collection, start=1):
    print(f"{idx}: {item}")