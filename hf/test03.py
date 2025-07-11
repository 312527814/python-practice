from transformers import BertTokenizer, BertForSequenceClassification,BertModel
from transformers import pipeline

# 加载模型和分词器
# model_name = r"D:\PycharmProjects\disanqi\demo_5\trasnFormers_test\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model_name = r"E:\BaiduNetdiskDownload\demo_13\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 创建分类 pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer,device="cpu")

# 进行分类
result = classifier("你好，我是一款语言模型")
print(result)
print(model)


