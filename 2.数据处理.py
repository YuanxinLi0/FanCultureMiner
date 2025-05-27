import pandas as pd
import re
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties

# 文件路径设置
csv_file = 'data/comments.csv'
fan_dict_path = 'tools/饭圈词典.txt'
stopwords_path = 'tools/stopword.txt'
wordcloud_output = 'output/2/wordcloud.png'
barchart_output = 'output/2/word_frequency.png'
output_file = 'data/processed_comments.csv'

# 1. 数据加载与文本清洗
df = pd.read_csv(csv_file, encoding='utf-8')
df.shape
df = df.dropna(subset=['text'])
df.shape

# 清洗文本
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
df = df[df['cleaned_text'].str.len() > 0]

# 2. 加载自定义词典和停用词
jieba.load_userdict(fan_dict_path)
with open(fan_dict_path, 'r', encoding='utf-8') as f:
    fan_words = [line.strip() for line in f.readlines()]
print(f"加载饭圈词典，共{len(fan_words)}个词")
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f.readlines()]
# 3. 文本分词
# 分词处理
df['segmented'] = df['cleaned_text'].apply(
    lambda x: ' '.join([w for w in jieba.cut(x) if w not in stopwords and len(w.strip()) >= 2])
)
# 4. 生成词频统计
all_words = ' '.join(df['segmented'].tolist())
word_list = all_words.split()
# 创建词频字典
words = {}
for word in word_list:
    if len(word) >= 2:
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1
word_counts = Counter(words)
# 5. 生成词频柱状图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(family='SimHei', size=11)
x = [item[0] for item in word_counts.most_common(10)]  # 取前10个词
y = [item[1] for item in word_counts.most_common(10)]  # 取对应的频次
plt.figure(figsize=(12, 6))
plt.grid(False)
bars = plt.bar(x, y, color='lightskyblue')
# 显示每个词的频次值
for i, v in enumerate(y):
    plt.text(i, v - max(y) * 0.05, f'{v}', ha='center')
# 设置标题和轴标签
plt.xlabel('关键词', fontproperties=font)
plt.ylabel('出现频次', fontproperties=font)
plt.title('饭圈评论高频词汇统计', fontproperties=font)
# 保存图形
plt.tight_layout()
plt.savefig(barchart_output, dpi=300)
plt.close()
# 6. 生成词云图
wordcloud = WordCloud(
    font_path='simhei.ttf',
    background_color="white",
    max_words=2000,
    width=800,
    height=800,
    max_font_size=150,
    random_state=42,
)
# 生成词云
wordcloud.fit_words(words)
# 保存词云图
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(wordcloud_output, dpi=300)
plt.close()
# 保存处理后的数据
df.to_csv(output_file, index=False, encoding='utf-8')
