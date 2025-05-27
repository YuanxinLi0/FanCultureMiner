import pandas as pd
df = pd.read_csv("data/processed_comments.csv")
df['datetime'] = pd.to_datetime(df['comment_time'], format='%Y-%m-%d %H:%M', errors='coerce')
df = df.dropna(subset=['datetime'])
daily_counts = df.groupby(df['datetime'].dt.date).size() # 按日期分组统计评论数量
# 查看重要时间点
print("评论数量最多的日期:")
print(daily_counts.nlargest(10))
monthly_counts = df.groupby([df['datetime'].dt.year, df['datetime'].dt.month]).size()
print("\n按月评论分布:")
print(monthly_counts)