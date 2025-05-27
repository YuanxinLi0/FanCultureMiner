import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import jieba
import warnings
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('TkAgg')


class TextDataset(Dataset):
    def __init__(self, sequences, labels, max_len=50):
        self.max_len = max_len
        self.sequences = [self.pad_sequence(seq) for seq in sequences]
        self.labels = torch.LongTensor(labels)

    def pad_sequence(self, seq):
        if len(seq) == 0:
            return np.zeros((self.max_len, 100))
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        padding = np.zeros((self.max_len - len(seq), 100))
        return np.vstack([seq, padding])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), self.labels[idx]


class Word2Vec_GRU(nn.Module):
    def __init__(self, embed_dim=100, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        output = gru_out[:, -1, :]
        output = self.dropout(output)
        return self.fc(output)


def load_and_process_data():
    print("正在加载和处理数据...")
    df = pd.read_csv('data/final_data_of_train.csv')
    df = df.dropna(subset=['cleaned_text', 'sentiment_category'])
    df = df[df['cleaned_text'].str.len() > 2]

    texts = df['cleaned_text'].tolist()
    labels = df['sentiment_category'].tolist()

    print(f"原始数据量: {len(texts)}")

    # 分词和创建序列
    tokenized_texts = []
    valid_labels = []

    for i, text in enumerate(tqdm(texts, desc="分词处理")):
        if ' ' in text:
            tokens = text.split()
        else:
            tokens = list(jieba.cut(text))

        tokens = [t.strip() for t in tokens if len(t.strip()) > 0]

        if len(tokens) >= 2:
            tokenized_texts.append(tokens)
            valid_labels.append(labels[i])

    print(f"有效文本数量: {len(tokenized_texts)}")

    # 训练Word2Vec
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5,
                     min_count=2, workers=4, sg=0, epochs=10)

    # 创建序列
    sequences = []
    final_labels = []

    for i, text in enumerate(tokenized_texts):
        seq = [model.wv[word] for word in text if word in model.wv]
        if len(seq) >= 1:
            sequences.append(np.array(seq))
            final_labels.append(valid_labels[i])

    print(f"最终数据量: {len(sequences)}")
    print(f"标签分布: {Counter(final_labels)}")

    return sequences, final_labels


def train_model(model, train_loader, val_loader, device, epochs=50):
    # 计算类别权重
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    class_counts = Counter(all_labels)
    total = len(all_labels)
    class_weights = torch.FloatTensor([
        total / (len(class_counts) * class_counts[0]),
        total / (len(class_counts) * class_counts[1])
    ]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for sequences, labels in tqdm(train_loader, desc=f'训练 {epoch + 1}/{epochs}'):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f'轮次 {epoch + 1}: 训练准确率={train_acc:.4f}, 验证准确率={val_acc:.4f}')

        # 自适应学习率
        if epoch > 5 and val_acc < max(val_accs[:-1]):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

    return train_losses, train_accs, val_losses, val_accs


def plot_curves(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(val_losses, 'r-', label='验证损失')
    plt.title('Word2vec + GRU 损失变化')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='训练准确率')
    plt.plot(val_accs, 'r-', label='验证准确率')
    plt.title('Word2vec + GRU 准确率变化')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.title('Word2vec + GRU 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载和处理数据
    sequences, labels = load_and_process_data()

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # 数据加载器
    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=32)

    # 模型
    model = Word2Vec_GRU().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("开始训练...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, device, epochs=50)

    # 绘制曲线
    plot_curves(train_losses, train_accs, val_losses, val_accs)

    # 测试
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.argmax(1).cpu().numpy())
            true_labels.extend(labels.numpy())

    test_accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\n测试准确率: {test_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(true_labels, predictions, target_names=['负面', '正面']))

    plot_confusion_matrix(true_labels, predictions)


if __name__ == "__main__":
    main()