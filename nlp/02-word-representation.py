# 加载公开词向量
from gensim.models import KeyedVectors

model_path = 'E:\\BaiduNetdiskDownload\\tl\\09_AI大模型之NLP教程\\3.代码\\代码\\text_rep\\data\\word2vec.txt'
model = KeyedVectors.load_word2vec_format(model_path)

# 1.维数
print(model.vector_size)

# 2.词数
print(len(model.index_to_key))

# 3.查看向量

print(model['地铁'])

# 4.相似度
print(model.similarity('地铁', '图书馆'))