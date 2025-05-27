import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取已标注的数据
df = pd.read_csv("./data/sentiment_processed_data.csv")
senlist = df['sentiment_score'].tolist()

amount = len(df)
datasetLen = (int(amount / 7) + 1)

# --------- 第一张图：情感分布 ---------
# 算一下平均值啥的
avg = np.mean(senlist)
med = np.median(senlist)
std = np.std(senlist)
n_pos = sum(1 for s in senlist if s > 0)
n_neg = sum(1 for s in senlist if s < 0)
n_mid = sum(1 for s in senlist if s == 0)
per_pos = n_pos / len(senlist) * 100
per_neg = n_neg / len(senlist) * 100
per_mid = n_mid / len(senlist) * 100

# 试试好看的颜色
colors = sns.color_palette("coolwarm", 100)

# 开始画图，做两层图的那种
fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [3, 1]})
ax1 = axes[0]

# 画直方图
bins = np.arange(-0.5, 0.51, 0.01)
hist_data, bin_edges = np.histogram(senlist, bins=bins)
bins_middle = (bin_edges[:-1] + bin_edges[1:]) / 2

# 染色
mycolors = []
for x in bins_middle:
    if x < 0:
        c = colors[int(abs(x) * 100)]
    else:
        c = colors[int(50 + x * 100)]
    mycolors.append(c)

bars = ax1.bar(bins_middle, hist_data, width=0.01, color=mycolors, edgecolor='none', alpha=0.9)

# 加一条曲线
kde = stats.gaussian_kde(senlist)
kde_x = np.linspace(-0.5, 0.5, 1000)
kde_y = kde(kde_x) * len(senlist) * 0.01
ax1.plot(kde_x, kde_y, '-', color='#333333', linewidth=2, label='平滑曲线')

# 加一些线
ax1.axvline(x=avg, color='black', linestyle='-', linewidth=2, label=f'均值 ({avg:.3f})')
ax1.axvline(x=med, color='blue', linestyle='--', linewidth=2, label=f'中位数 ({med:.3f})')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='中性')

# 标准差范围
ax1.axvspan(avg - std, avg + std, alpha=0.2, color='gray', label=f'±1标准差 ({std:.3f})')

# 图的标题和轴标签
ax1.set_title('微博评论情感分布', fontsize=15, pad=15, fontweight='bold')
ax1.set_xlabel('情感值 [-0.5超负面 到 0.5超正面]', fontsize=12)
ax1.set_ylabel('数量', fontsize=12)

# 加图例
ax1.legend(fontsize=10)

# 加个小文本框解释一下
txt = (f"总评论: {len(senlist):,}\n"
      f"均值: {avg:.3f}\n"
      f"中位数: {med:.3f}\n"
      f"标准差: {std:.3f}\n"
      f"正面评论: {per_pos:.1f}%\n"
      f"负面评论: {per_neg:.1f}%")

ax1.text(0.02, 0.97, txt, transform=ax1.transAxes,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 设置范围和刻度
ax1.set_xlim(-0.5, 0.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# 下面画个饼图
ax2 = axes[1]
labels = ['正面', '中性', '负面']
sizes = [n_pos, n_mid, n_neg]
explode = (0.1, 0, 0.1)  # 凸显正负面
colors = ['#d73027', '#f7f7f7', '#4575b4']

wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels,
                                  autopct='%1.1f%%', colors=colors,
                                  textprops={'fontsize': 12})

ax2.set_title('情感比例', fontsize=14)

# 调整布局保存
plt.tight_layout()
plt.savefig('./output/3.2+3.3/sentiment_distribution_sci.png', dpi=300)
plt.close()

# --------- 第二张图：情感波动 ---------
# 用滚动平均让图不那么乱
wsize = min(1000, len(senlist) // 20)
rmean = pd.Series(senlist).rolling(window=wsize, center=True).mean()

# 画图
fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111)

# 加个渐变背景
for i in range(0, 101, 20):
    ax.axhspan(-0.5 + i/100, -0.5 + (i+20)/100,
              color=f'C{i//20}', alpha=0.05)

# 画原始数据点和平均线
xr = np.arange(0, len(senlist), 1)
ax.plot(xr, senlist, 'k.', alpha=0.03, markersize=1, label='原始数据')
ax.plot(xr, rmean, '-', color='#e41a1c', linewidth=2.5, label=f'{wsize}点滚动平均')

# 加中性线和平均线
ax.axhline(y=0, color='blue', linestyle='--', linewidth=1.5, label='中性线')
ax.axhline(y=avg, color='green', linestyle='-', linewidth=1.5, label=f'平均值={avg:.3f}')

# 简单拟合一条趋势线
z = np.polyfit(xr, pd.Series(senlist).fillna(0), 1)
trend = z[0] * xr + z[1]
ax.plot(xr, trend, 'g--', linewidth=2, label=f'趋势线 (斜率={z[0]:.7f})')

# 加区域
sections = [
    (0, len(senlist)//7, "2024年6月下旬"),
    (len(senlist)//7, 2*len(senlist)//7, "2024年7月"),
    (2*len(senlist)//7, 3*len(senlist)//7, "2024年8月-2025年4月"),
    (3*len(senlist)//7, 4*len(senlist)//7, "2025年5月第一周"),
    (4*len(senlist)//7, 5*len(senlist)//7, "2025年5月第二周"),
    (5*len(senlist)//7, 6*len(senlist)//7, "2025年5月第三周"),
    (6*len(senlist)//7, len(senlist), "2025年5月下旬")
]

random.seed(42)  # 固定随机种子
for start, end, name in sections:
    # 根据时段类型应用不同颜色
    if "2024年6月" in name:
        color = '#ff9999'  # 红色系表示第一个高峰
    elif "2025年5月" in name:
        color = '#99ff99'  # 绿色系表示第二个高峰
    else:
        color = '#cccccc'  # 灰色表示沉寂期
    ax.axvspan(start, end, color=color, alpha=0.2)
    mid = (start + end) // 2
    ax.annotate(name, xy=(mid, 0.4), ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 标题和标签
ax.set_title('微博评论情感波动图', fontsize=16, fontweight='bold')
ax.set_xlabel('评论序号', fontsize=13)
ax.set_ylabel('情感值 [-0.5到0.5]', fontsize=13)

# 图例
ax.legend(loc='upper right', fontsize=10)

# 网格
ax.grid(True, linestyle='--', alpha=0.3)

# 保存
plt.tight_layout()
plt.savefig('./output/3.2+3.3/sentiment_fluctuation_sci.png', dpi=300)
plt.close()

# --------- 第三张图：情感变化趋势 ---------
# 计算时间段的情感值
result_score = []
for i in range(7):
    start_idx = i * datasetLen
    end_idx = min((i + 1) * datasetLen, amount)
    if start_idx < amount:
        segment_sentiment = df.iloc[start_idx:end_idx]['sentiment_score'].sum()
        result_score.append(segment_sentiment)
print("时间段情感值:", result_score)

# 时间标签
time_labels = [
    "2024年6月下旬",
    "2024年7月",
    "2024年8月-\n2025年4月",
    "2025年5月\n第1周",
    "2025年5月\n第2周",
    "2025年5月\n第3周",
    "2025年5月\n下旬"
]

# 标签匹配
if len(time_labels) > len(result_score):
    time_labels = time_labels[:len(result_score)]
elif len(time_labels) < len(result_score):
    time_labels.extend([f"时段{i+len(time_labels)+1}" for i in range(len(result_score) - len(time_labels))])

# 每段的评论数
cnt = []
for i in range(len(result_score)):
    start_idx = i * datasetLen
    end_idx = min((i + 1) * datasetLen, amount)
    if start_idx < amount:
        cnt.append(end_idx - start_idx)

# 做单轴图
fig, ax1 = plt.subplots(figsize=(12, 8))

# 背景搞点颜色
fig.patch.set_facecolor('#fafafa')
ax1.set_facecolor('#fafafa')

# 画情感值柱状图
x = range(len(result_score))
bars = ax1.bar(x, result_score, width=0.6, color='#4393c3', edgecolor='#2166ac',
              linewidth=1.5, alpha=0.8, label='情感值')

# 加个虚线
ax1.axhline(y=0, color='#d6604d', linestyle='--', linewidth=2, label='0线')

# 加数值标签
for i, (bar, val) in enumerate(zip(bars, result_score)):
    height = bar.get_height()
    offset = 15 if height >= 0 else -25
    ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, offset), textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

# 设置标签
ax1.set_xlabel('时间段', fontsize=14, fontweight='bold')
ax1.set_ylabel('情感值', fontsize=14, color='#4393c3')

# 标题
ax1.set_title('舆情变化趋势', fontsize=18, fontweight='bold')

# X轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(time_labels, rotation=30, ha='right', fontsize=11)

# 网格
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

# 图例
ax1.legend(loc='upper left', fontsize=12)

# 调整并保存
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('./output/3.2+3.3/sentiment_trend_sci.png', dpi=300)
plt.close()