
from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('LLM-Research/Llama-3.2-1B-Instruct-evals',cache_dir='F:\llm\dataset')
#
#
# print(ds)
#
# for data in ds:
#     print(data)
#
# for data in enumerate(ds.get("latest")):
#     print(data[1])
#
# print(next(iter(ds)))

#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('xeon09112/ptb_text_only', subset_name='default', split='validation',cache_dir='F:\llm\dataset')

#您可按需配置 subset_name、split，参照“快速使用”示例代码
print(ds)

for data in ds:
    print(data)

for data in enumerate(ds):
    print(data[1])

# print(next(iter(ds)))