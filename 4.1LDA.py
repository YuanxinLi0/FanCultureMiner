import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import jieba
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from snownlp import SnowNLP
import re
import pyLDAvis
import pyLDAvis.gensim_models

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv('data/final_data_of_train.csv')
comments_data = data['cleaned_text']
comments = comments_data[comments_data.apply(len) >= 4]

print(f"原始评论数量: {len(comments_data)}")
print(f"过滤后评论数量: {len(comments)}")

coms = data['sentiment_score']
pos_data = data[data['sentiment_category'] == 1]['segmented'].dropna()
neg_data = data[data['sentiment_category'] == 0]['segmented'].dropna()

print(f"正面评论数量: {len(pos_data)}")
print(f"负面评论数量: {len(neg_data)}")

# 去停用词
stop = pd.read_csv('./tools/stopword.txt', sep='bunzaizai', encoding='utf-8', header=None)
stop = ['', ' '] + list(stop[0])

# 创建数据框并处理停用词
pos = pd.DataFrame(pos_data)
neg = pd.DataFrame(neg_data)

pos[1] = pos['segmented'].apply(lambda s: s.split(' '))
pos[2] = pos[1].apply(lambda x: [i for i in x if i not in stop])

neg[1] = neg['segmented'].apply(lambda s: s.split(' '))
neg[2] = neg[1].apply(lambda x: [i for i in x if i not in stop])

print("去停用词完成")

# 定义计算主题一致性得分的函数
def coherence(num_topics, corpus=None, dictionary=None, texts=None):
    """计算主题一致性得分"""
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
    return coherence_model.get_coherence()

# 正面主题分析
pos_dict = corpora.Dictionary(pos[2])
pos_corpus = [pos_dict.doc2bow(i) for i in pos[2]]

# 使用简化的可视化方式绘制正面评论主题一致性图
print("=== 正面评论主题数量优化 ===")
x = range(2, 9)  # 从2到8个主题
y = [coherence(i, corpus=pos_corpus, dictionary=pos_dict, texts=pos[2]) for i in x]

# 记录最优主题数
optimal_pos_topics = x[y.index(max(y))]
print(f"正面评论最优主题数量: {optimal_pos_topics}")

# 绘制正面评论主题一致性图
plt.figure(figsize=(12, 10))
plt.plot(x, y)
plt.xlabel('主题数目')
plt.ylabel('coherence大小')
plt.title('正面评论主题-coherence变化情况')
plt.savefig('./output/4.1/pos_coherence_plot.png', dpi=300)
plt.show()

# 训练正面LDA模型
pos_lda = models.LdaModel(pos_corpus, num_topics=optimal_pos_topics, id2word=pos_dict, random_state=42, passes=10)

for i in range(optimal_pos_topics):
    print(f'pos_topic{i}')
    print(pos_lda.print_topic(i))
    print()

# 正面LDA可视化
print("生成正面评论LDA可视化...")
pos_vis = pyLDAvis.gensim_models.prepare(pos_lda, pos_corpus, pos_dict)
pyLDAvis.save_html(pos_vis, './output/4.1/pos_lda_visualization.html')
print("正面评论LDA可视化已保存到 pos_lda_visualization.html")

# 负面主题分析
neg_dict = corpora.Dictionary(neg[2])
neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]]

# 使用简化的可视化方式绘制负面评论主题一致性图
print("=== 负面评论主题数量优化 ===")
x = range(2, 9)  # 从2到8个主题
y = [coherence(i, corpus=neg_corpus, dictionary=neg_dict, texts=neg[2]) for i in x]

# 记录最优主题数
optimal_neg_topics = x[y.index(max(y))]
print(f"负面评论最优主题数量: {optimal_neg_topics}")

# 绘制负面评论主题一致性图
plt.figure(figsize=(12, 10))
plt.plot(x, y)
plt.xlabel('主题数目')
plt.ylabel('coherence大小')
plt.title('负面评论主题-coherence变化情况')
plt.savefig('./output/4.1/neg_coherence_plot.png', dpi=300)
plt.show()

# 训练负面LDA模型
neg_lda = models.LdaModel(neg_corpus, num_topics=optimal_neg_topics, id2word=neg_dict, random_state=42, passes=10)

for i in range(optimal_neg_topics):
    print(f'neg_topic{i}')
    print(neg_lda.print_topic(i))
    print()

# 负面LDA可视化
print("生成负面评论LDA可视化...")
neg_vis = pyLDAvis.gensim_models.prepare(neg_lda, neg_corpus, neg_dict)
pyLDAvis.save_html(neg_vis, './output/4.1/neg_lda_visualization.html')
pos_lda.save('./output/4.1/pos_lda_model')
neg_lda.save('./output/4.1/neg_lda_model')
print("模型已保存")

# 绘制情感得分趋势变化图
amount = len(coms)
datasetLen = (int(amount / 7) + 1)

# 计算每个时间段的平均情感值
segment_scores = []
time_labels = [
    "2024年6月下旬",
    "2024年7月",
    "2024年8月-2025年4月",
    "2025年5月第1周",
    "2025年5月第2周",
    "2025年5月第3周",
    "2025年5月下旬"
]

for i in range(7):
    start_idx = i * datasetLen
    end_idx = min((i + 1) * datasetLen, amount)
    if start_idx < amount:
        segment_data = coms.iloc[start_idx:end_idx]
        avg_score = segment_data.mean()
        segment_scores.append(avg_score)

# 绘制分段趋势图
x = range(len(segment_scores))
plt.figure(figsize=(12, 10))
plt.plot(x, segment_scores)
plt.xlabel('时间段')
plt.ylabel('平均情感得分')
plt.title('评论情感趋势变化图')
plt.xticks(x, time_labels, rotation=45, ha='right')
plt.savefig('./output/4.1/lda_sentiment_trend.png', dpi=300)
plt.show()

# 综合统计信息
print("\n=== 综合分析结果 ===")
print(f"总评论数量: {len(comments)}")
print(f"正面评论数量: {len(pos_data)} ({len(pos_data) / len(comments) * 100:.1f}%)")
print(f"负面评论数量: {len(neg_data)} ({len(neg_data) / len(comments) * 100:.1f}%)")
print(f"正面评论最优主题数: {optimal_pos_topics}")
print(f"负面评论最优主题数: {optimal_neg_topics}")
print("所有分析结果已保存到 ./output/ 目录下")