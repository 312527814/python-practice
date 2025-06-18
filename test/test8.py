# data = [('physics',1), ('chemistry',2), ('chemistry3',3), ('chemistry4',4)]
# sents = [i[0] for i in data]
# print(sents)

# for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data):


def collate_fn(data):

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]


    return input_ids, attention_mask, token_type_ids

# if __name__ == '__main__':
#     data1={"input_ids":1,"attention_mask":2,"token_type_ids":3}
#     data2={"input_ids":12,"attention_mask":22,"token_type_ids":32}
#     data3={"input_ids":13,"attention_mask":23,"token_type_ids":33}
#
#     datas=[data1,data2,data3]
#     for i, data in enumerate(datas):
#         print(i)
#         print(data)

fruits = ['apple', 'banana', 'orange']

# print(type(fruits))
# for index, fruit in enumerate(fruits):
#     print(index, fruit)
#
# out=[1,0,1,0]
# labels=[1,0,1,0]
# a= (out==labels).sum().item()/len(labels)
#
# print(a)




import torch

# 假设 out 是模型预测的结果，labels 是真实标签
out = torch.tensor([2, 0, 1, 2])
labels = torch.tensor([2, 1, 1, 0])

# 比较预测是否正确
correct = (out == labels)
print(correct)  # 输出: tensor([ True, False,  True, False])