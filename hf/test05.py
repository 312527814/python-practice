from transformers import BertTokenizer, BertForSequenceClassification,BertModel
from transformers import pipeline

# 加载模型和分词器
# model_name = r"D:\PycharmProjects\disanqi\demo_5\trasnFormers_test\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model_name = r"E:\BaiduNetdiskDownload\demo_13\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model = BertForSequenceClassification.from_pretrained(model_name)

model2 = BertModel.from_pretrained(model_name)

print(model)

print("................................")

print(model2)


