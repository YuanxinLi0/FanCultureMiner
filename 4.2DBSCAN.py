import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv('./data/sentiment_processed_data.csv')

# 使用jieba进行中文分词
data['segmented'] = data['cleaned_text'].apply(lambda x: ' '.join(jieba.lcut(x)))

# TF-IDF向量化
vectorizer_word = TfidfVectorizer(max_features=2000, token_pattern=r'(?u)\b\w+\b', min_df=5, analyzer='word',
                                  ngram_range=(1, 2))
tfidf_matrix = vectorizer_word.fit_transform(data['segmented'])

# DBSCAN聚类
clustering = DBSCAN(eps=0.95, min_samples=5).fit(tfidf_matrix)
n_clusters = len(pd.Series(clustering.labels_).value_counts())
labels = clustering.labels_
cluster_counts = pd.Series(clustering.labels_).value_counts()

print(f"估计的聚类个数为: {n_clusters}")
print("各群样本数量:")
print(cluster_counts)

# 将聚类结果添加到原始数据中
data['cluster_labels'] = clustering.labels_

# === 意见领袖分析 ===
print("\n=== 意见领袖分析 ===")

# 使用真实的社交媒体指标计算影响力
data['text_length'] = data['cleaned_text'].str.len()
data['sentiment_strength'] = abs(data['sentiment_score'])  # 情感强度
data['influence_score'] = (data['like_count'] * 0.5 +
                           data['reply_count'] * 2 +
                           data['text_length'] * 0.01 +
                           data['sentiment_strength'] * 20 * 0.2)

# 在每个聚类中找出意见领袖（影响力最高的前3名）
opinion_leaders = []
for cluster_id in data['cluster_labels'].unique():
    if cluster_id == -1:  # 跳过噪声点
        continue

    cluster_data = data[data['cluster_labels'] == cluster_id]
    if len(cluster_data) < 3:  # 群体太小跳过
        continue

    # 前3名
    top_leaders = cluster_data.nlargest(3, 'influence_score')

    print(f"\n群体 {cluster_id} 的意见领袖 (共{len(cluster_data)}人):")
    for idx, leader in top_leaders.iterrows():
        print(f"  用户: {leader.get('user_name', 'Unknown')}")
        print(
            f"  影响力得分: {leader['influence_score']:.2f} (点赞:{leader['like_count']}, 回复:{leader['reply_count']})")
        print(f"  评论内容: {leader['cleaned_text'][:50]}...")
        print(f"  情感得分: {leader['sentiment_score']:.3f}")
        print()

    opinion_leaders.extend(top_leaders.index.tolist())


pca = PCA(n_components=2)
tfidf_matrix_dense = tfidf_matrix.toarray()
reduced_data = pca.fit_transform(tfidf_matrix_dense)

noise_mask = labels == -1
non_noise_mask = labels != -1

# 创建更大的画布
plt.figure(figsize=(14, 10))

# 绘制噪声点
plt.scatter(reduced_data[noise_mask, 0], reduced_data[noise_mask, 1],
            c='lightgray', marker='x', alpha=0.4, s=25, label='噪声点', zorder=1)

# 获取所有非噪声群体
unique_labels = np.unique(labels[non_noise_mask])
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

# 分类显示群体：只为主要群体（人数>15）显示详细图例
major_clusters = []
medium_clusters = []
small_clusters = []

for i, label in enumerate(unique_labels):
    if label == -1:
        continue

    mask = labels == label
    cluster_size = np.sum(mask)

    if cluster_size > 50:  # 大群体
        major_clusters.append((label, cluster_size))
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                    c=[colors[i]], label=f'群体{label}({cluster_size}人)',
                    alpha=0.7, s=35, zorder=2)
    elif cluster_size > 15:  # 中等群体
        medium_clusters.append((label, cluster_size))
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                    c=[colors[i]], alpha=0.6, s=25, zorder=2)
    else:  # 小群体
        small_clusters.append((label, cluster_size))
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                    c=[colors[i]], alpha=0.4, s=15, zorder=2)

# 为中等群体和小群体添加汇总图例项
if medium_clusters:
    plt.scatter([], [], c='steelblue', alpha=0.6, s=25,
                label=f'中等群体({len(medium_clusters)}个,16-50人)')
if small_clusters:
    plt.scatter([], [], c='lightblue', alpha=0.4, s=15,
                label=f'小群体({len(small_clusters)}个,<16人)')

# 突出显示意见领袖 - 使用更醒目的标记
if opinion_leaders:
    leader_points = reduced_data[opinion_leaders]
    plt.scatter(leader_points[:, 0], leader_points[:, 1],
                c='red', marker='*', s=200, edgecolors='darkred',
                linewidth=2, label='意见领袖', alpha=0.95, zorder=5)

# 优化图例布局 - 分两列放在右侧，避免遮挡数据
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
           frameon=True, fancybox=True, shadow=True, fontsize=10)

# 美化图表
plt.title('DBSCAN聚类结果与意见领袖分布', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('主成分1 (PCA降维)', fontsize=12)
plt.ylabel('主成分2 (PCA降维)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# 添加统计信息文本框
info_text = f'总聚类数: {n_clusters}\n大群体: {len(major_clusters)}个\n中等群体: {len(medium_clusters)}个\n小群体: {len(small_clusters)}个\n噪声点: {np.sum(noise_mask)}个\n意见领袖: {len(opinion_leaders)}位'
plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=9)

plt.tight_layout()
plt.savefig('./output/4.2/dbscan_opinion_leaders_improved.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# === 意见领袖统计分析 ===
print(f"\n总共发现 {len(opinion_leaders)} 位意见领袖")
leader_data = data.loc[opinion_leaders]
print(f"意见领袖平均点赞数: {leader_data['like_count'].mean():.1f}")
print(f"意见领袖平均回复数: {leader_data['reply_count'].mean():.1f}")
print(f"意见领袖平均影响力得分: {leader_data['influence_score'].mean():.2f}")
print(f"意见领袖平均情感得分: {leader_data['sentiment_score'].mean():.3f}")
print(f"意见领袖情感分布:")
print(f"  正面: {(leader_data['sentiment_score'] > 0.1).sum()}人")
print(f"  中性: {((leader_data['sentiment_score'] >= -0.1) & (leader_data['sentiment_score'] <= 0.1)).sum()}人")
print(f"  负面: {(leader_data['sentiment_score'] < -0.1).sum()}人")

# 输出群体分类统计
print(f"\n群体分类统计:")
print(f"大群体（>50人）: {len(major_clusters)}个")
for label, size in major_clusters:
    print(f"  群体{label}: {size}人")
print(f"中等群体（16-50人）: {len(medium_clusters)}个")
print(f"小群体（<16人）: {len(small_clusters)}个")
print(f"噪声点: {np.sum(noise_mask)}个")

# === 模型评估 ===
print('\n模型评估:')
print('同质性: %.3f' % metrics.homogeneity_score(data['sentiment_category'], data['cluster_labels']))
print('完整性: %.3f' % metrics.completeness_score(data['sentiment_category'], data['cluster_labels']))
print('同质性和完整性的调和平均: %.3f' % metrics.v_measure_score(data['sentiment_category'], data['cluster_labels']))
print('调整兰德指数: %.3f' % metrics.adjusted_rand_score(data['sentiment_category'], data['cluster_labels']))
print('调整互信息: %.3f' % metrics.adjusted_mutual_info_score(data['sentiment_category'], data['cluster_labels']))

print("\n可视化图表已保存:")
print("1. ./output/4.2/dbscan_opinion_leaders_improved.png - DBSCAN聚类与意见领袖分布图")