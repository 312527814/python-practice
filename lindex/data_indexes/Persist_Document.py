# import
#pip install llama-index-vector-stores-chroma
#pip install llama-index-embeddings-huggingface
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display



#初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbedding(
    #指定了一个预训练的sentence-transformer模型的路径
    model_name="/root/autodl-tmp/mode/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# #将创建的嵌入模型赋值给全局设置的embed_model属性，这样在后续的索引构建过程中，就会使用这个模型
# Settings.embed_model = embed_model

# load documents
documents = SimpleDirectoryReader("/root/autodl-tmp/data/").load_data()



index = VectorStoreIndex.from_documents(
    documents,embed_model=embed_model
)

index.storage_context.persist()
