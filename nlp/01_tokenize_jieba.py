import jieba


## 第一节 分词模式
text = "小明毕业于北京大学计算机系"
ss = jieba.cut(text)
# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(ss))  # 全模式
cut_all_full = jieba.lcut(text, cut_all=True)
print(cut_all_full)
cut_all = jieba.lcut(text)
print(cut_all)



## 第二节 自定义词典
jieba.lo
jieba.load_userdict("./data/user_dict.txt")