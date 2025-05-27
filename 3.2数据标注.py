import pandas as pd
from snownlp import SnowNLP
# 读csv文件
df = pd.read_csv("./data/processed_comments.csv")
senlist = []
for line in df['cleaned_text']:
    s = SnowNLP(line)
    # 情感值转一下
    senlist.append(s.sentiments - 0.5)
print(f"总共有{len(senlist)}条评论")
# 处理时间
df['datetime'] = pd.to_datetime(df['comment_time'], format='%Y-%m-%d %H:%M', errors='coerce')
df = df.dropna(subset=['datetime']).sort_values('datetime')
# 把情感值放到数据里面
df['sentiment_score'] = senlist[:len(df)]
df['sentiment_category'] = [1 if s > 0 else 0 for s in df['sentiment_score']]
df.to_csv('./data/sentiment_processed_data.csv', index=False)
