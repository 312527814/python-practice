from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 设置具体包含 config.json 的目录
model_dir = r"E:\BaiduNetdiskDownload\demo_13\TransFormers_test\my_model_cache\uer\gpt2-chinese-cluecorpussmall\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
import transformers
import torch



# 使用加载的模型和分词器创建生成文本的 pipeline
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer,device="cpu")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=1000,
    tokenizer=tokenizer
)

from langchain import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline = pipeline,
                          model_kwargs = {'temperature':0.7,'max_length':50,'truncation':True,'top_k':50,'top_p':0.9})

template = "{input}"

# 使用模板创建提示
from langchain import PromptTemplate, LLMChain
prompt=PromptTemplate.from_template(template)
# 创建LLMChain实例

llm_chain = LLMChain(prompt=prompt, llm=llm)

response= llm_chain.run("请介绍一下你是谁？")

print(response)


