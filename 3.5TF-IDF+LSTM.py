import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import matplotlib

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            model.load_state_dict(self.best_weights)
            return True
        return False


class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TFIDF_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=150, hidden_dim=64, num_classes=2, seq_len=25):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.feature_dim = embed_dim // seq_len  # 150 // 25 = 6

        # TF-IDF特征投影到嵌入空间
        self.feature_proj = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 双向LSTM
        self.lstm = nn.LSTM(
            self.feature_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # 投影TF-IDF特征到嵌入空间
        x = self.feature_proj(x)  # [batch_size, embed_dim]

        # 重塑为序列格式以便LSTM处理
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.feature_dim)  # [batch_size, seq_len, feature_dim]

        # LSTM处理序列
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]

        # 使用平均池化聚合序列特征
        x = torch.mean(lstm_out, dim=1)  # [batch_size, hidden_dim*2]

        # 分类
        return self.classifier(x)


def load_data():
    print("正在加载数据...")
    df = pd.read_csv('./data/final_data_of_train.csv')
    texts = df['cleaned_text'].fillna('').tolist()
    labels = df['sentiment_category'].tolist()
    print(f"数据量: {len(texts)}")
    return texts, labels


def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=10)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

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

            # 梯度裁剪
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
        avg_val_loss = val_loss / len(val_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_acc = val_acc / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        print(f'轮次 {epoch + 1}: 训练损失={avg_train_loss:.4f}, 验证损失={avg_val_loss:.4f}, '
              f'训练准确率={avg_train_acc:.4f}, 验证准确率={avg_val_acc:.4f}')

        scheduler.step(avg_val_loss)

        # 早停检查
        if early_stopping(avg_val_loss, model):
            print(f'早停触发，在第{epoch + 1}轮停止训练')
            break

    return train_losses, train_accs, val_losses, val_accs


def plot_curves(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.title('TF-IDF + LSTM: 损失变化')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率', color='blue')
    plt.plot(val_accs, label='验证准确率', color='red')
    plt.title('TF-IDF + LSTM: 准确率变化')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.title('TF-IDF + LSTM: 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    texts, labels = load_data()

    print("创建TF-IDF特征...")
    # 使用TF-IDF提取特征
    vectorizer = TfidfVectorizer(
        max_features=2500,  # 适中的特征数量
        ngram_range=(1, 2),  # 1-2gram
        min_df=2,  # 最小文档频率
        max_df=0.95,  # 最大文档频率
        stop_words=None  # 不使用停用词，保留更多信息
    )

    features = vectorizer.fit_transform(texts).toarray()
    print(f"TF-IDF特征维度: {features.shape[1]}")

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # 数据加载器
    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=64)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=64)

    # 模型初始化
    model = TFIDF_LSTM(vocab_size=features.shape[1]).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    print("开始训练...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, device)

    # 绘制训练曲线
    plot_curves(train_losses, train_accs, val_losses, val_accs)

    # 测试评估
    print("开始测试评估...")
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.argmax(1).cpu().numpy())
            true_labels.extend(labels.numpy())

    # 输出结果
    test_accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\n测试准确率: {test_accuracy:.4f}")

    print("\n详细分类报告:")
    print(classification_report(true_labels, predictions, target_names=['负面', '正面']))

    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions)


if __name__ == "__main__":
    main()