from modelscope.msdatasets import MsDataset

# # 以cats_and_dogs数据集为例，数据集链接： https://modelscope.cn/datasets/tany0699/cats_and_dogs/summary
# ds = MsDataset.load('cats_and_dogs', namespace='tany0699', split='train')
# print(next(iter(ds)))
#
# # 也可以通过namespace/dataset_name的形式传入数据集名称
# ds = MsDataset.load('tany0699/cats_and_dogs', split='train')
# print(next(iter(ds)))
#
# # 使用强制加载模式（删除该数据集的本地缓存并强制重新下载）
# ds = MsDataset.load('cats_and_dogs', namespace='tany0699', split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
# print(next(iter(ds)))



from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('LLM-Research/Llama-3.2-1B-Instruct-evals',cache_dir='F:\llm\dataset')


print(ds)

for data in ds:
    print(data)

# for data in enumerate(ds.get("latest")):
#     print(data[1])

# print(next(iter(ds)))