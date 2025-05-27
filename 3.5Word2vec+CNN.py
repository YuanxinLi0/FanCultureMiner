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
import matplotlib
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TextDataset(Dataset):
    def __init__(self, sequences, labels, max_len=50):
        self.max_len = max_len
        self.sequences = [self.pad_sequence(seq) for seq in sequences]
        self.labels = torch.LongTensor(labels)

    def pad_sequence(self, seq):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        padding = np.zeros((self.max_len - len(seq), seq.shape[1] if len(seq) > 0 else 100))
        return np.vstack([seq, padding]) if len(seq) > 0 else padding

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), self.labels[idx]


class Word2Vec_CNN(nn.Module):
    def __init__(self, embed_dim=100, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 4, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        return self.fc(x)


def load_data():
    print("正在加载数据...")
    df = pd.read_csv('data/final_data_of_train.csv')
    texts = df['cleaned_text'].fillna('').tolist()
    labels = df['sentiment_category'].tolist()

    print("正在分词...")
    tokenized_texts = []
    for text in texts:
        tokens = text.split() if text.strip() else list(jieba.cut(text))
        tokenized_texts.append([t for t in tokens if t.strip()])

    print(f"数据量: {len(tokenized_texts)}")
    return tokenized_texts, labels


def create_word2vec_sequences(tokenized_texts, labels):
    print("训练Word2Vec模型...")
    filtered_texts = [text for text in tokenized_texts if text]
    model = Word2Vec(sentences=filtered_texts, vector_size=100, window=5, min_count=2, workers=4, sg=1, epochs=50)

    print("转换为词向量序列...")
    sequences = []
    for text in tokenized_texts:
        seq = [model.wv[word] for word in text if word in model.wv]
        sequences.append(np.array(seq) if seq else np.zeros((1, 100)))

    return sequences


def train_model(model, train_loader, val_loader, device, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_acc = 0, 0

        train_bar = tqdm(train_loader, desc=f'轮次 {epoch + 1}/{epochs} [训练]')
        for sequences, labels in train_bar:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).float().mean().item()

            train_bar.set_postfix({
                '损失': f'{loss.item():.4f}',
                '准确率': f'{100 * train_acc / (train_bar.n + 1):.2f}%'
            })

        # 验证阶段
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item()
                val_acc += (outputs.argmax(1) == labels).float().mean().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        print(f'轮次 {epoch + 1}: 训练损失={avg_train_loss:.4f}, 训练准确率={avg_train_acc:.4f}, '
              f'验证损失={avg_val_loss:.4f}, 验证准确率={avg_val_acc:.4f}')

    return train_losses, train_accs, val_losses, val_accs


def plot_curves(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.title('Word2Vec + CNN: 损失变化')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率', color='blue')
    plt.plot(val_accs, label='验证准确率', color='red')
    plt.title('Word2Vec + CNN: 准确率变化')
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
    plt.title('Word2Vec + CNN: 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenized_texts, labels = load_data()
    sequences = create_word2vec_sequences(tokenized_texts, labels)

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=32)

    model = Word2Vec_CNN().to(device)
    print("开始训练...")
    train_losses, train_accs, val_losses, val_accs = train_model(model, train_loader, val_loader, device)

    # 绘制训练曲线
    plot_curves(train_losses, train_accs, val_losses, val_accs)

    # 测试评估
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.argmax(1).cpu().numpy())
            true_labels.extend(labels.numpy())

    print("\n分类报告:")
    print(classification_report(true_labels, predictions, target_names=['负面', '正面']))
    print(f"测试准确率: {np.mean(np.array(predictions) == np.array(true_labels)):.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions)


if __name__ == "__main__":
    main()