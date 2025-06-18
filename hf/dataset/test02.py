from datasets import load_dataset,load_from_disk


dataset =load_dataset(path="csv", data_files=f"E:/BaiduNetdiskDownload/demo_7/data/news/train.csv", split="train")
print(dataset)
#查看数据
for data in dataset:
    print(data)