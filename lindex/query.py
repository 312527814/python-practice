from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM


#使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(model_name="/root/autodl-tmp/mode/Qwen/Qwen1___5-1___8B-Chat",
               tokenizer_name="/root/autodl-tmp/mode/Qwen/Qwen1___5-1___8B-Chat",
               model_kwargs={"trust_remote_code":True},
               tokenizer_kwargs={"trust_remote_code":True})

# 1. 初始化你的嵌入模型（必须与保存时一致）
embed_model = HuggingFaceEmbedding(
    model_name="/root/autodl-tmp/mode/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. 创建存储上下文并指定持久化目录
storage_context = StorageContext.from_defaults(persist_dir="./storage")

# 3. 使用load_index_from_storage从存储上下文中加载索引
index = load_index_from_storage(storage_context, embed_model=embed_model)

# 4. 获取查询引擎
query_engine = index.as_query_engine(llm=llm)

# 5. 开始查询
response = query_engine.query("xtuner是什么？")
print(response)