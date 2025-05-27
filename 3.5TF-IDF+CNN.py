import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import matplotlib

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ImprovedTFIDF_CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_classes=2):  # 减小嵌入维度
        super().__init__()
        self.proj = nn.Linear(vocab_size, embed_dim)

        # 简化CNN结构，减少参数
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)  # 减少通道数
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)  # 减少通道数，统一kernel size

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)

        # 增强正则化
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)  # 增加dropout
        self.batch_norm1 = nn.BatchNorm1d(32)  # 添加batch normalization
        self.batch_norm2 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.proj(x)).unsqueeze(1)  # 早期dropout

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout1(x)  # 卷积后dropout

        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout2(x)  # 更强的dropout

        x = self.pool(x).squeeze(2)
        return self.fc(x)


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def load_data():
    print("正在加载数据...")
    df = pd.read_csv('data/final_data_of_train.csv')
    texts = df['segmented'].fillna('').tolist()
    labels = df['sentiment_category'].tolist()
    print(f"数据量: {len(texts)}")
    print(f"数据量: {len(texts)}")
    return texts, labels


def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.CrossEntropyLoss()
    # 降低学习率，添加权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # 早停机制
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_acc = 0, 0

        train_bar = tqdm(train_loader, desc=f'轮次 {epoch + 1}/{epochs} [训练]')
        for features, labels in train_bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
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

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 早停检查
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"早停触发，在第{epoch + 1}轮停止训练")
            break

    return train_losses, train_accs, val_losses, val_accs


def plot_curves(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.title('TF-IDF + CNN: 损失变化')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率', color='blue')
    plt.plot(val_accs, label='验证准确率', color='red')
    plt.title('TF-IDF + CNN: 准确率变化')
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
    plt.title('TF-IDF + CNN: 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    texts, labels = load_data()

    print("创建TF-IDF特征...")
    # 减少特征数量，降低模型复杂度
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    features = vectorizer.fit_transform(texts).toarray()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 增大batch size，有助于稳定训练
    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=128)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=128)

    model = ImprovedTFIDF_CNN(features.shape[1]).to(device)
    print("开始训练...")
    train_losses, train_accs, val_losses, val_accs = train_model(model, train_loader, val_loader, device)

    # 绘制训练曲线
    plot_curves(train_losses, train_accs, val_losses, val_accs)

    # 测试评估
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.argmax(1).cpu().numpy())
            true_labels.extend(labels.numpy())

    print("\n分类报告:")
    print(classification_report(true_labels, predictions, target_names=['负面', '正面']))
    print(f"测试准确率: {np.mean(np.array(predictions) == np.array(true_labels)):.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions)


if __name__ == "__main__":
    main()