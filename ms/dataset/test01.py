from modelscope.hub.snapshot_download import snapshot_download


# model_dir = snapshot_download('Qwen/QwQ-32B-GGUF',local_dir='F:/llm')

#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('modelscope/hellaswag', subset_name='default', split='train',cache_dir='F:/llm')
#您可按需配置 subset_name、split，参照“快速使用”示例代码

# from modelscope import MsDataset
# ds_dict = MsDataset.load('squad')
# print(ds['train'][0])
print(ds)
#
# print(ds.keys())

#查看数据
for data in ds:
    print(data)
