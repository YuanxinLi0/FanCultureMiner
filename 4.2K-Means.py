import matplotlib
matplotlib.use('TkAgg')
import jieba
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv('./data/sentiment_processed_data.csv')
data1 = pd.DataFrame()
data1[0] = data['sentiment_category']
data1[1] = data['cleaned_text']

# 划分训练集和测试集
train_data, test_data = train_test_split(data1, test_size=0.2, random_state=42)


# 文本预处理函数
def seg_word(data):
    corpus = []
    stop = pd.read_csv('./tools/stopword.txt', sep='bunzaizai', encoding='utf-8', header=None)
    stopwords = [' '] + list(stop[0])
    for i in range(len(data)):
        string = data.iloc[i, 1].strip()
        seg_list = jieba.cut(string, cut_all=False)
        corpu = []
        for word in seg_list:
            if word not in stopwords:
                corpu.append(word)
        corpus.append(' '.join(corpu))
    return corpus


# 特征提取
vectorizer = TfidfVectorizer(max_features=2000)
train_corpus = seg_word(train_data)
test_corpus = seg_word(test_data)
train_tfidf = vectorizer.fit_transform(train_corpus)
test_tfidf = vectorizer.transform(test_corpus)
train_weight = train_tfidf.toarray()
test_weight = test_tfidf.toarray()

# K-Means 聚类
clf = KMeans(n_clusters=2, random_state=42)
clf.fit(train_weight)

print('2个中心点为：' + str(clf.cluster_centers_))
joblib.dump(clf, './output/4.2/km.pkl')

# === 意见领袖分析 ===
print("\n=== K-Means意见领袖分析 ===")

# 使用真实的社交媒体指标
# 需要将原始数据的指标添加到训练数据中
original_data = pd.read_csv('./data/sentiment_processed_data.csv')
train_indices = train_data.index
train_data_copy = train_data.copy()
train_data_copy['cluster_labels'] = clf.labels_

# 添加真实的社交媒体指标
train_data_copy['like_count'] = original_data.loc[train_indices, 'like_count'].values
train_data_copy['reply_count'] = original_data.loc[train_indices, 'reply_count'].values
train_data_copy['user_name'] = original_data.loc[train_indices, 'user_name'].values
train_data_copy['text_length'] = train_data_copy[1].str.len()

# 计算影响力得分
train_data_copy['influence_score'] = (train_data_copy['like_count'] * 0.5 +
                                      train_data_copy['reply_count'] * 0.3 +
                                      train_data_copy['text_length'] * 0.01 * 0.2)

# 在每个聚类中找出意见领袖
opinion_leaders = []
for cluster_id in [0, 1]:  # K=2的聚类
    cluster_data = train_data_copy[train_data_copy['cluster_labels'] == cluster_id]

    # 按影响力排序，取前5名作为意见领袖
    top_leaders = cluster_data.nlargest(5, 'influence_score')

    sentiment_label = "正面" if cluster_id == 1 else "负面"
    print(f"\n{sentiment_label}群体 (群体{cluster_id}) 的意见领袖 (共{len(cluster_data)}人):")
    for idx, leader in top_leaders.iterrows():
        print(f"  用户: {leader['user_name']}")
        print(
            f"  影响力得分: {leader['influence_score']:.2f} (点赞:{leader['like_count']}, 回复:{leader['reply_count']})")
        print(f"  评论内容: {leader[1][:50]}...")
        print()

    opinion_leaders.extend(top_leaders.index.tolist())

# 聚类结果可视化降维
pca = PCA(n_components=2)
newData = pca.fit_transform(train_weight)

labels = clf.labels_.tolist()
plt.figure(figsize=(10, 8))
label_colors = ['#008000', '#FF0000']
color = [label_colors[i] for i in labels]
plt.scatter(newData[:, 0], newData[:, 1], c=color, alpha=0.5)

# 突出显示意见领袖
leader_indices = [i for i, idx in enumerate(train_data_copy.index) if idx in opinion_leaders]
leader_points = newData[leader_indices]
plt.scatter(leader_points[:, 0], leader_points[:, 1],
            c='black', marker='*', s=200, edgecolors='yellow',
            linewidth=2, label='意见领袖', alpha=1.0)

plt.title('K-Means聚类结果与意见领袖 (K=2)')
plt.legend()
plt.savefig('./output/4.2/kmeans_opinion_leaders.png')
plt.show()

# 意见领袖统计
leader_data = train_data_copy.loc[opinion_leaders]
print(f"\n总共发现 {len(opinion_leaders)} 位意见领袖")
print(f"意见领袖平均影响力得分: {leader_data['influence_score'].mean():.2f}")
print(f"正面群体意见领袖: {(leader_data['cluster_labels'] == 1).sum()}人")
print(f"负面群体意见领袖: {(leader_data['cluster_labels'] == 0).sum()}人")

# 原有模型评估
cluster_labels = np.zeros_like(clf.labels_)
if np.mean(clf.labels_ == 1) > 0.5:
    cluster_labels[clf.labels_ == 1] = 0
    cluster_labels[clf.labels_ == 0] = 1
else:
    cluster_labels[clf.labels_ == 0] = 0
    cluster_labels[clf.labels_ == 1] = 1

train_acc = np.mean(cluster_labels == train_data[0].values)
print(f'\n训练集准确率: {train_acc:.4f}')

test_pred = clf.predict(test_weight)
test_cluster_labels = np.zeros_like(test_pred)
if np.mean(clf.labels_ == 1) > 0.5:
    test_cluster_labels[test_pred == 1] = 0
    test_cluster_labels[test_pred == 0] = 1
else:
    test_cluster_labels[test_pred == 0] = 0
    test_cluster_labels[test_pred == 1] = 1

test_acc = np.mean(test_cluster_labels == test_data[0].values)
print(f'测试集准确率: {test_acc:.4f}')