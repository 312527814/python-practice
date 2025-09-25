# 导入transformers库中的AdamW优化器
from torch.optim import AdamW
# 导入transformers库中的优化器调度器获取函数
from transformers.optimization import get_scheduler
# 导入PyTorch库
import torch
# 导入自定义的数据集类MyDataset
from data import MyDataset
# 导入transformers库中的自动分词器
from transformers import AutoTokenizer
# 导入transformers库中的因果语言模型和GPT2模型
from transformers import AutoModelForCausalLM, GPT2Model

# 实例化自定义数据集
dataset = MyDataset()

mode_path=r"E:\BaiduNetdiskDownload\huggingface-model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"

# 加载预训练的编码器（分词器）
tokenizer = AutoTokenizer.from_pretrained(mode_path)
# 加载预训练的模型
model = AutoModelForCausalLM.from_pretrained(mode_path)
# 打印模型结构（已注释）
# print(model)

# 定义数据预处理函数，用于将文本编码成模型所需的格式
def collate_fn(data):
    # 使用分词器对数据进行编码，并添加必要的填充和截断
    data = tokenizer.batch_encode_plus(data,
                                       padding=True,  # 填充序列
                                       truncation=True,  # 截断序列
                                       max_length=512,  # 最大序列长度
                                       return_tensors='pt')  # 返回PyTorch张量

    # 创建标签，与输入ID相同
    data['labels'] = data['input_ids'].clone()

    return data

# 创建数据加载器，用于批量加载数据
loader = torch.utils.data.DataLoader(
    dataset=dataset,  # 指定数据集
    batch_size=2,  # 指定批量大小
    collate_fn=collate_fn,  # 指定预处理函数
    shuffle=True,  # 打乱数据
    drop_last=True,  # 如果最后一个批次不足，则丢弃
)
# 打印数据加载器中的批次数量
print(len(loader))

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义训练函数
def train():
    global model  # 使用全局变量model
    # 确定使用CPU还是GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 将模型移动到指定设备
    model = model.to(DEVICE)

    print("............1")
    print(type(model.parameters()))

    for i, data in enumerate(model.parameters()):
        print(data)
    print("............2")

    # 实例化优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # 获取优化器调度器
    scheduler = get_scheduler(name='linear',  # 线性调度器
                              num_warmup_steps=0,  # 预热步数
                              num_training_steps=len(loader),  # 总训练步数
                              optimizer=optimizer)

    # 设置模型为训练模式
    model.train()
    # 遍历数据加载器中的每个批次
    for i, data in enumerate(loader):
        # 将数据移动到指定设备
        for k in data.keys():
            data[k] = data[k].to(device)
        # 前向传播
        out = model(**data)
        # 获取损失
        loss = out['loss']

        # 反向传播
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新模型参数
        optimizer.step()
        # 更新学习率
        scheduler.step()

        # 清空梯度
        optimizer.zero_grad()
        model.zero_grad()

        # 每隔50个批次打印一次训练信息
        if i % 50 == 0:
            # 准备标签和输出用于计算准确率
            labels = data['labels'][:, 1:]
            #通过‘logits’获取模型的原始输出值
            out = out['logits'].argmax(dim=2)[:, :-1]

            # 移除在数据预处理阶段添加的填充（通常是0），以便只计算实际数据部分的损失和准确率，避免填充部分对模型性能评估的影响。
            select = labels != 0
            labels = labels[select]
            out = out[select]
            del select

            # 计算准确率
            accuracy = (labels == out).sum().item() / labels.numel()

            # 获取当前学习率
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            # 打印批次索引、损失、学习率和准确率
            print(i, loss.item(),lr, accuracy)
    # 保存模型参数，不保存模型结构
    torch.save(model.state_dict(), 'net.pt')
    # 打印模型参数保存成功信息
    print("权重保存成功！")

# 当脚本作为主程序运行时，执行以下代码
if __name__ == '__main__':
    # 进行1000个训练周期
    for epoch in range(1000):
        # 调用训练函数
        train()
