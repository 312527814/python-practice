# import
#pip install llama-index-vector-stores-chroma
#pip install llama-index-embeddings-huggingface
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser



#初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbedding(
    #指定了一个预训练的sentence-transformer模型的路径
    model_name="/root/autodl-tmp/mode/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# load documents
documents = SimpleDirectoryReader("/root/autodl-tmp/data/").load_data()
#创建节点解析器
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
#将文档分割成节点
base_node = node_parser.get_nodes_from_documents(documents=documents)

print(base_node)

#根据自定义的node节点构建向量索引
index = VectorStoreIndex(nodes=base_node,embed_model=embed_model)


index.storage_context.persist(persist_dir="./persist/node/")
