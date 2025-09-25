from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 设置具体包含 config.json 的目录
model_dir = r"E:\BaiduNetdiskDownload\demo_13\TransFormers_test\my_model_cache\uer\gpt2-chinese-cluecorpussmall\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
import transformers
import torch



# 使用加载的模型和分词器创建生成文本的 pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer,device="cpu")
result = generator("请介绍一下你是谁？")
print(result)



