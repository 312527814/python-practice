import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm
import requests
import zipfile
import io
import os

# ================= 1. 下载同义词数据集 =================
print("正在下载同义词数据集...")


def download_synonym_data():
    """下载同义词数据集"""
    # 使用中文同义词词林扩展版
    # 如果没有本地数据，这里是一个示例数据集
    synonym_pairs = [
        # (词1, 词2, 相似度得分 1.0=完全同义, 0.0=不同义)
        ("高兴", "快乐", 1.0),
        ("高兴", "愉快", 0.9),
        ("高兴", "兴奋", 0.7),
        ("悲伤", "难过", 1.0),
        ("悲伤", "伤心", 0.9),
        ("悲伤", "痛苦", 0.8),
        ("美丽", "漂亮", 1.0),
        ("美丽", "好看", 0.9),
        ("美丽", "秀丽", 0.8),
        ("大", "巨大", 0.8),
        ("大", "庞大", 0.7),
        ("小", "微小", 0.8),
        ("小", "细小", 0.7),
        ("学习", "读书", 0.8),
        ("学习", "研究", 0.7),
        ("工作", "职业", 0.8),
        ("工作", "劳动", 0.7),
        ("汽车", "轿车", 0.9),
        ("汽车", "车辆", 0.8),
        ("电脑", "计算机", 1.0),
        ("手机", "电话", 0.8),
        ("快速", "迅速", 0.9),
        ("快速", "敏捷", 0.8),
        ("慢", "缓慢", 0.9),
        ("热", "炎热", 0.9),
        ("冷", "寒冷", 0.9),
        ("好", "优秀", 0.8),
        ("坏", "糟糕", 0.8),
        ("爱", "喜欢", 0.8),
        ("恨", "讨厌", 0.8),
        ("父亲", "爸爸", 1.0),
        ("母亲", "妈妈", 1.0),
        ("学生", "学员", 0.8),
        ("老师", "教师", 1.0),
        ("医院", "诊所", 0.7),
        ("学校", "学院", 0.8),
        ("城市", "都市", 0.9),
        ("乡村", "农村", 0.9),
        ("河流", "江河", 0.9),
        ("山", "山脉", 0.8),
        ("海", "海洋", 0.9),
        ("书", "书籍", 0.9),
        ("写", "书写", 0.9),
        ("说", "讲话", 0.8),
        ("听", "聆听", 0.8),
        ("看", "观看", 0.9),
        ("吃", "食用", 0.8),
        ("喝", "饮用", 0.9),
        ("走", "行走", 0.9),
        ("跑", "奔跑", 0.9),
        ("跳", "跳跃", 0.9),
        ("买", "购买", 0.9),
        ("卖", "出售", 0.9),
        ("开始", "启动", 0.8),
        ("结束", "终止", 0.8),
        ("帮助", "协助", 0.9),
        ("成功", "胜利", 0.8),
        ("失败", "失利", 0.8),
    ]

    # 添加反义词作为负样本
    antonym_pairs = [
        ("高兴", "悲伤", 0.0),
        ("大", "小", 0.0),
        ("好", "坏", 0.0),
        ("爱", "恨", 0.0),
        ("热", "冷", 0.0),
        ("快", "慢", 0.0),
        ("买", "卖", 0.3),  # 不完全反义，有一定关系
        ("开始", "结束", 0.0),
        ("成功", "失败", 0.0),
    ]

    return synonym_pairs + antonym_pairs


# ================= 2. 构建词表和数据处理器 =================
class SynonymDatasetProcessor:
    """同义词数据集处理器"""

    def __init__(self, synonym_pairs):
        self.synonym_pairs = synonym_pairs
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self._build_vocab()

    def _build_vocab(self):
        """构建词表"""
        # 收集所有词
        all_words = set()
        for word1, word2, _ in self.synonym_pairs:
            all_words.add(word1)
            all_words.add(word2)

        # 构建映射
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

        for idx, word in enumerate(sorted(all_words), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)

        print(f"词汇表大小: {self.vocab_size}")
        print(f"词汇示例: {list(self.word2idx.keys())[:20]}")

        # 统计同义词对
        synonym_count = len([s for s in self.synonym_pairs if s[2] > 0.5])
        antonym_count = len(self.synonym_pairs) - synonym_count
        print(f"同义词对: {synonym_count}, 反义词对: {antonym_count}")

    def get_word_index(self, word):
        """获取词索引"""
        return self.word2idx.get(word, self.word2idx['<UNK>'])

    def get_training_samples(self):
        """获取训练样本"""
        samples = []
        for word1, word2, similarity in self.synonym_pairs:
            idx1 = self.get_word_index(word1)
            idx2 = self.get_word_index(word2)

            # 正样本（相似度高）
            if similarity >= 0.7:
                label = 1.0  # 相似
                samples.append((idx1, idx2, label, 1.0))  # 权重

            # 负样本（相似度低）
            elif similarity <= 0.3:
                label = 0.0  # 不相似
                samples.append((idx1, idx2, label, 1.0))

            # 中等相似度（作为soft label）
            else:
                label = similarity  # 实际相似度
                samples.append((idx1, idx2, label, 0.5))  # 较低权重

        return samples


# ================= 3. 创建PyTorch Dataset =================
class SynonymEmbeddingDataset(Dataset):
    """同义词Embedding训练数据集"""

    def __init__(self, processor):
        self.processor = processor
        self.samples = processor.get_training_samples()

        print(f"训练样本数量: {len(self.samples)}")

        # 显示一些示例
        print("\n训练样本示例:")
        for i in range(min(5, len(self.samples))):
            idx1, idx2, label, weight = self.samples[i]
            word1 = processor.idx2word[idx1]
            word2 = processor.idx2word[idx2]
            print(f"  {word1} - {word2}: 标签={label:.2f}, 权重={weight:.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx1, idx2, label, weight = self.samples[idx]
        return {
            'word1': torch.tensor(idx1, dtype=torch.long),
            'word2': torch.tensor(idx2, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32)
        }


# ================= 4. 定义语义Embedding模型 =================
class SemanticEmbeddingModel(nn.Module):
    """语义Embedding模型"""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 共享的Embedding层
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, word_indices):
        """前向传播"""
        return self.embeddings(word_indices)

    def compute_similarity(self, idx1, idx2):
        """计算两个词的余弦相似度"""
        vec1 = self.embeddings(idx1)
        vec2 = self.embeddings(idx2)

        # 余弦相似度
        cos_sim = F.cosine_similarity(vec1, vec2, dim=-1)
        return cos_sim


# ================= 5. 训练函数 =================
def train_semantic_embeddings(model, dataset, num_epochs=500, learning_rate=0.01):
    """训练语义Embedding"""

    # 数据加载器
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 损失函数：均方误差
    criterion = nn.MSELoss(reduction='none')  # 不立即求平均

    print("\n开始训练语义Embedding...")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            word1 = batch['word1']
            word2 = batch['word2']
            labels = batch['label']
            weights = batch['weight']

            # 前向传播
            optimizer.zero_grad()

            # 获取词向量
            vec1 = model(word1)
            vec2 = model(word2)

            # 计算余弦相似度作为预测值
            pred_similarity = F.cosine_similarity(vec1, vec2, dim=-1)

            # 计算损失（带权重）
            loss = criterion(pred_similarity, labels)
            weighted_loss = (loss * weights).mean()

            # 反向传播
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item() * len(word1)
            total_samples += len(word1)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

            # 验证一些词对
            model.eval()
            with torch.no_grad():
                # 测试同义词
                test_pairs = [("高兴", "快乐"), ("悲伤", "难过"),
                              ("大", "小"), ("好", "坏")]

                print("  测试相似度:")
                for w1, w2 in test_pairs:
                    idx1 = torch.tensor([dataset.processor.get_word_index(w1)])
                    idx2 = torch.tensor([dataset.processor.get_word_index(w2)])
                    sim = model.compute_similarity(idx1, idx2).item()
                    print(f"    {w1} - {w2}: {sim:.4f}")

            model.train()

    return model


# ================= 6. 主程序 =================
def main():
    print("=" * 60)
    print("语义Embedding训练系统")
    print("=" * 60)

    # 1. 下载同义词数据
    synonym_data = download_synonym_data()

    # 2. 处理数据
    processor = SynonymDatasetProcessor(synonym_data)

    # 3. 创建数据集
    dataset = SynonymEmbeddingDataset(processor)

    # 4. 创建模型
    vocab_size = processor.vocab_size
    embed_dim = 10  # 向量维度
    model = SemanticEmbeddingModel(vocab_size, embed_dim)

    print(f"\n模型参数:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  向量维度: {embed_dim}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 5. 训练前验证（随机权重）
    print("\n训练前相似度（随机权重）:")
    test_words = ["高兴", "快乐", "悲伤", "大", "小"]
    model.eval()
    with torch.no_grad():
        for word in test_words:
            if word in processor.word2idx:
                idx = torch.tensor([processor.word2idx[word]])
                vec = model.embeddings(idx)[0].numpy()
                print(f"  {word}: {vec[:5].round(4)}...")  # 显示前5维

    # 6. 训练模型
    trained_model = train_semantic_embeddings(
        model=model,
        dataset=dataset,
        num_epochs=200,
        learning_rate=0.02
    )

    # 7. 训练后验证
    print("\n" + "=" * 60)
    print("训练结果验证")
    print("=" * 60)

    trained_model.eval()
    with torch.no_grad():
        # 测试相似词对
        test_cases = [
            ("高兴", "快乐", "同义词（应接近1.0）"),
            ("高兴", "愉快", "近义词（应>0.7）"),
            ("高兴", "悲伤", "反义词（应接近0.0）"),
            ("大", "小", "反义词（应接近0.0）"),
            ("美丽", "漂亮", "同义词（应接近1.0）"),
            ("电脑", "计算机", "同义词（应接近1.0）"),
            ("汽车", "车辆", "相关词（应>0.5）"),
        ]

        print("\n词对相似度测试:")
        for w1, w2, desc in test_cases:
            if w1 in processor.word2idx and w2 in processor.word2idx:
                idx1 = torch.tensor([processor.word2idx[w1]])
                idx2 = torch.tensor([processor.word2idx[w2]])
                sim = trained_model.compute_similarity(idx1, idx2).item()
                print(f"  {w1:4} - {w2:4}: {sim:.4f}  ({desc})")

    # 8. 查找相似词
    print("\n查找相似词（基于向量余弦相似度）:")

    def find_similar_words(model, target_word, top_k=5):
        """查找与目标词最相似的词"""
        if target_word not in processor.word2idx:
            print(f"  词汇表中没有 '{target_word}'")
            return

        target_idx = processor.word2idx[target_word]
        target_vec = model.embeddings(torch.tensor([target_idx]))[0]

        similarities = []
        for word, idx in processor.word2idx.items():
            if word not in ['<PAD>', '<UNK>'] and word != target_word:
                word_vec = model.embeddings(torch.tensor([idx]))[0]
                sim = F.cosine_similarity(target_vec.unsqueeze(0),
                                          word_vec.unsqueeze(0)).item()
                similarities.append((word, sim))

        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"  与'{target_word}'最相似的词:")
        for word, sim in similarities[:top_k]:
            print(f"    {word}: {sim:.4f}")

    # 测试几个词
    test_targets = ["高兴", "悲伤", "大", "学习", "汽车"]
    for word in test_targets:
        find_similar_words(trained_model, word, top_k=3)

    # 9. 保存Embedding层
    print("\n保存Embedding层...")
    embedding_weights = trained_model.embeddings.weight.data.numpy()

    # 创建可用的Embedding层
    final_embedding = nn.Embedding.from_pretrained(
        torch.FloatTensor(embedding_weights),
        freeze=False  # 可以继续训练
    )

    print(f"Embedding层已创建:")
    print(f"  形状: {final_embedding.weight.shape}")
    print(f"  示例 - '高兴'的向量:")
    idx = processor.word2idx.get("高兴", processor.word2idx['<UNK>'])
    vec = final_embedding(torch.tensor([idx])).detach().numpy()[0]
    print(f"    {vec.round(4)}")

    return final_embedding, processor


# ================= 7. 运行训练 =================
if __name__ == "__main__":
    # 运行训练
    trained_embedding, processor = main()

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    # 使用示例
    print("\n使用示例:")

    word = '高兴'
    word_idx = torch.tensor([processor.word2idx[word]])
    word_vector = trained_embedding(word_idx)
    print(f'{word}的向量: {word_vector}')

    word2 = '快乐'
    word_idx2 = torch.tensor([processor.word2idx[word2]])
    word_vector2 = trained_embedding(word_idx2)
    print(f'{word2}的向量: {word_vector2}')

    similarity = F.cosine_similarity(word_vector, word_vector2, dim=-1)
    print(f'{word}和{word2}余弦相似度是：{similarity.item()}')