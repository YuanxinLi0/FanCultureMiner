import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import jieba
import re
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置页面配置
st.set_page_config(
    page_title="微博饭圈文化舆情分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 缓存数据加载函数
@st.cache_data
def load_data(file_path):
    """加载CSV数据"""
    return pd.read_csv(file_path, encoding='utf-8')


# 缓存模型加载
@st.cache_resource
def load_models():
    models = {}
    from gensim import models as gensim_models
    models['pos_lda'] = gensim_models.LdaModel.load('output/4.1/pos_lda_model')
    models['neg_lda'] = gensim_models.LdaModel.load('output/4.1/neg_lda_model')
    import joblib
    models['kmeans'] = joblib.load('output/4.2/km.pkl')
    return models


# 侧边栏
with st.sidebar:
    st.title("🎯 功能导航")

    # 功能选择
    page = st.selectbox(
        "选择功能模块",
        ["数据概览", "数据预处理", "情感分析", "主题建模",
         "聚类分析", "社交网络分析", "综合报告"]
    )

    st.markdown("---")

    # 数据源选择
    st.subheader("📁 数据源")
    data_source = st.radio(
        "选择数据文件",
        ["原始评论", "处理后数据", "情感标注数据", "最终训练数据"]
    )

    # 映射文件路径
    file_mapping = {
        "原始评论": "data/comments.csv",
        "处理后数据": "data/processed_comments.csv",
        "情感标注数据": "data/sentiment_processed_data.csv",
        "最终训练数据": "data/final_data_of_train.csv"
    }

    # 加载数据
    if os.path.exists(file_mapping[data_source]):
        df = load_data(file_mapping[data_source])
        st.success(f"✅ 已加载 {len(df)} 条数据")
    else:
        st.error("❌ 数据文件不存在")
        df = None

# 主界面标题
st.title("📊 微博饭圈文化舆情分析系统")
st.markdown("---")
# 数据概览页面
if page == "数据概览":
    st.header("📈 数据概览")

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总评论数", f"{len(df):,}")

        with col2:
            if 'user_name' in df.columns:
                st.metric("用户数", f"{df['user_name'].nunique():,}")

        with col3:
            if 'sentiment_score' in df.columns:
                avg_sentiment = df['sentiment_score'].mean()
                st.metric("平均情感分数", f"{avg_sentiment:.3f}")

        with col4:
            if 'like_count' in df.columns:
                total_likes = df['like_count'].sum()
                st.metric("总点赞数", f"{total_likes:,}")
        # 数据预览
        st.subheader("📋 数据预览")
        # 列选择
        if len(df.columns) > 5:
            selected_cols = st.multiselect(
                "选择要显示的列",
                df.columns.tolist(),
                default=df.columns[:5].tolist()
            )
            if selected_cols:
                st.dataframe(df[selected_cols].head(100))
        else:
            st.dataframe(df.head(100))
        # 基本统计
        st.subheader("📊 基本统计")
        st.write(df.describe())
        # 时间分布
        if 'comment_time' in df.columns:
            st.subheader("📅 时间分布")
            df['datetime'] = pd.to_datetime(df['comment_time'], errors='coerce')
            df['date'] = df['datetime'].dt.date

            daily_counts = df.groupby('date').size()

            fig, ax = plt.subplots(figsize=(12, 6))
            daily_counts.plot(kind='bar', ax=ax)
            ax.set_title('每日评论数量分布')
            ax.set_xlabel('日期')
            ax.set_ylabel('评论数')
            plt.xticks(rotation=45)
            st.pyplot(fig)

# 数据预处理页面
elif page == "数据预处理":
    st.header("🔧 数据预处理")

    if df is not None:
        # 预处理选项
        st.subheader("⚙️ 预处理选项")

        col1, col2 = st.columns(2)

        with col1:
            clean_text = st.checkbox("文本清洗", value=True)
            segment_text = st.checkbox("中文分词", value=True)
            remove_stopwords = st.checkbox("去除停用词", value=True)

        with col2:
            sentiment_analysis = st.checkbox("情感分析", value=True)
            remove_duplicates = st.checkbox("去除重复", value=True)

        if st.button("🚀 开始预处理"):
            with st.spinner("正在处理..."):
                progress_bar = st.progress(0)
                # 文本清洗
                if clean_text and 'text' in df.columns:
                    progress_bar.progress(20)
                    st.info("正在进行文本清洗...")

                    def clean_text_func(text):
                        if not isinstance(text, str):
                            return ""
                        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
                        text = re.sub(r'<.*?>', '', text)
                        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
                        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text

                    df['cleaned_text'] = df['text'].apply(clean_text_func)
                # 分词
                if segment_text and 'cleaned_text' in df.columns:
                    progress_bar.progress(40)
                    st.info("正在进行中文分词...")
                    # 加载停用词
                    with open('tools/stopword.txt', 'r', encoding='utf-8') as f:
                        stopwords = set([line.strip() for line in f])
                    def segment_func(text):
                        words = jieba.lcut(text)
                        if remove_stopwords:
                            words = [w for w in words if w not in stopwords and len(w.strip()) >= 2]
                        return words
                    df['segmented'] = df['cleaned_text'].apply(segment_func)
                    df['segmented_text'] = df['segmented'].apply(lambda x: ' '.join(x))

                # 情感分析
                if sentiment_analysis:
                    progress_bar.progress(60)
                    st.info("正在进行情感分析...")
                    from snownlp import SnowNLP

                    df['sentiment_score'] = df['cleaned_text'].apply(
                        lambda x: SnowNLP(x).sentiments if x else 0
                    )
                    # 情感分类规则
                    df['sentiment_category'] = df['sentiment_score'].apply(
                        lambda x: 1 if x > 0 else (0 if x < 0 else 2)  # 大于0为积极，小于0为消极，等于0为中性
                    )

                # 去重
                if remove_duplicates and 'cleaned_text' in df.columns:
                    progress_bar.progress(80)
                    st.info("正在去除重复数据...")
                    original_len = len(df)
                    df = df.drop_duplicates(subset=['cleaned_text'], keep='first')
                    st.success(f"去除了 {original_len - len(df)} 条重复数据")

                progress_bar.progress(100)
                st.success("✅ 预处理完成！")

                # 显示结果
                st.subheader("📊 预处理结果")
                st.dataframe(df.head(50))

                # 保存选项
                if st.button("💾 保存处理后的数据"):
                    output_path = f"data/preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(output_path, index=False, encoding='utf-8')
                    st.success(f"数据已保存到: {output_path}")

# 情感分析页面
elif page == "情感分析":
    st.header("💭 情感分析")

    if df is not None:
        # 检查并确定情感分数列
        sentiment_col = None
        sentiment_display_name = None

        # 检查可能的情感分数列
        if 'sentiment_score' in df.columns:
            sentiment_col = 'sentiment_score'
            sentiment_display_name = '情感分数'
        elif 'sentiment_category' in df.columns:
            # 如果只有类别，可以基于类别创建简单的分数
            st.info("未找到情感分数列，基于情感类别生成分数...")
            df['sentiment_score_generated'] = df['sentiment_category'].map({0: 0.2, 1: 0.8, 2: 0.5})
            sentiment_col = 'sentiment_score_generated'
            sentiment_display_name = '生成的情感分数'
        else:
            st.warning("⚠️ 当前数据中没有情感分析结果，请先在'数据预处理'中进行情感分析")
            st.stop()

        # 确保情感分数在0-1范围内
        if df[sentiment_col].min() < 0:
            # 如果是centered版本（-0.5到0.5），转换回0-1
            df['sentiment_score_normalized'] = df[sentiment_col] + 0.5
            sentiment_col = 'sentiment_score_normalized'
            sentiment_display_name = '标准化情感分数'

        # 情感分布统计
        col1, col2, col3 = st.columns(3)

        with col1:
            positive_ratio = (df[sentiment_col] > 0.6).mean()
            st.metric("正面比例", f"{positive_ratio:.1%}")

        with col2:
            neutral_ratio = ((df[sentiment_col] >= 0.4) & (df[sentiment_col] <= 0.6)).mean()
            st.metric("中性比例", f"{neutral_ratio:.1%}")

        with col3:
            negative_ratio = (df[sentiment_col] < 0.4).mean()
            st.metric("负面比例", f"{negative_ratio:.1%}")

        # 情感分布图
        st.subheader("📊 情感分布可视化")

        col1, col2 = st.columns(2)

        with col1:
            # 情感分数直方图
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df[sentiment_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(df[sentiment_col].mean(), color='red', linestyle='--',
                       label=f'平均值: {df[sentiment_col].mean():.3f}')
            ax.set_xlabel(sentiment_display_name)
            ax.set_ylabel('频次')
            ax.set_title(f'{sentiment_display_name}分布')
            ax.legend()
            st.pyplot(fig)

        with col2:
            # 情感类别饼图
            sentiment_counts = pd.Series({
                '正面': (df[sentiment_col] > 0.6).sum(),
                '中性': ((df[sentiment_col] >= 0.4) & (df[sentiment_col] <= 0.6)).sum(),
                '负面': (df[sentiment_col] < 0.4).sum()
            })

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#4CAF50', '#FFC107', '#F44336']
            ax.pie(sentiment_counts.values, labels=sentiment_counts.index,
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('情感类别分布')
            st.pyplot(fig)

        # 时间趋势分析
        time_col = None
        if 'datetime' in df.columns:
            time_col = 'datetime'
        elif 'comment_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['comment_time'], errors='coerce')
            time_col = 'datetime'

        if time_col and df[time_col].notna().any():
            st.subheader("📈 情感时间趋势")

            # 按日期聚合
            df['date'] = pd.to_datetime(df[time_col]).dt.date
            daily_sentiment = df.groupby('date')[sentiment_col].agg(['mean', 'count'])

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # 情感均值
            ax1.plot(daily_sentiment.index, daily_sentiment['mean'],
                     'b-', marker='o', label='平均情感分数')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('平均情感分数', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax1.legend(loc='upper left')

            # 评论数量
            ax2 = ax1.twinx()
            ax2.bar(daily_sentiment.index, daily_sentiment['count'],
                    alpha=0.3, color='green', label='评论数量')
            ax2.set_ylabel('评论数量', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.legend(loc='upper right')

            plt.title('情感趋势与评论量变化')
            fig.autofmt_xdate()
            st.pyplot(fig)

        # 情感分析详情
        st.subheader("📝 情感分析详情")

        # 选择情感类别查看
        sentiment_type = st.selectbox(
            "选择要查看的情感类别",
            ["全部", "正面", "中性", "负面"]
        )

        if sentiment_type == "正面":
            filtered_df = df[df[sentiment_col] > 0.6]
        elif sentiment_type == "中性":
            filtered_df = df[(df[sentiment_col] >= 0.4) & (df[sentiment_col] <= 0.6)]
        elif sentiment_type == "负面":
            filtered_df = df[df[sentiment_col] < 0.4]
        else:
            filtered_df = df

        # 显示样例
        text_col = None
        if 'text' in filtered_df.columns:
            text_col = 'text'
        elif 'cleaned_text' in filtered_df.columns:
            text_col = 'cleaned_text'

        if text_col:
            st.write(f"共 {len(filtered_df)} 条{sentiment_type}评论")

            # 随机显示一些评论
            sample_size = min(10, len(filtered_df))
            if sample_size > 0:
                samples = filtered_df.sample(n=sample_size)
                for _, row in samples.iterrows():
                    with st.expander(f"情感分数: {row[sentiment_col]:.3f}"):
                        st.write(row[text_col])
                        if 'user_name' in row:
                            st.caption(f"用户: {row['user_name']}")
                        if 'like_count' in row:
                            st.caption(f"👍 {row['like_count']} | 💬 {row.get('reply_count', 0)}")

# 主题建模页面
elif page == "主题建模":
    st.header("🎯 主题建模")

    # 检查已有的LDA结果
    if os.path.exists('output/4.1/pos_lda_visualization.html'):
        st.subheader("📊 已有主题建模结果")

        tab1, tab2 = st.tabs(["正面评论主题", "负面评论主题"])

        with tab1:
            st.write("正面评论LDA主题分析")

            # 显示HTML可视化
            with open('output/4.1/pos_lda_visualization.html', 'r', encoding='utf-8') as f:
                html_content = f.read()

            st.components.v1.html(html_content, height=800, scrolling=True)

            # 显示一致性图
            if os.path.exists('output/4.1/pos_coherence_plot.png'):
                st.image('output/4.1/pos_coherence_plot.png',
                         caption='正面评论主题一致性分析')

        with tab2:
            st.write("负面评论LDA主题分析")

            # 显示HTML可视化
            with open('output/4.1/neg_lda_visualization.html', 'r', encoding='utf-8') as f:
                html_content = f.read()

            st.components.v1.html(html_content, height=800, scrolling=True)

            # 显示一致性图
            if os.path.exists('output/4.1/neg_coherence_plot.png'):
                st.image('output/4.1/neg_coherence_plot.png',
                         caption='负面评论主题一致性分析')

    # 新建主题建模
    st.subheader("🆕 新建主题模型")

    if df is not None and 'segmented' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            num_topics = st.slider("主题数量", 2, 10, 5)
            sentiment_filter = st.selectbox(
                "选择情感类别",
                ["全部", "正面", "负面"]
            )

        with col2:
            max_features = st.slider("最大特征数", 100, 1000, 500)
            iterations = st.slider("迭代次数", 10, 50, 20)

        if st.button("🎯 开始主题建模"):
            with st.spinner("正在进行主题建模..."):
                try:
                    from gensim import corpora, models

                    # 过滤数据
                    if 'sentiment_score' in df.columns:
                        sentiment_col = 'sentiment_score'
                    elif 'sentiment_category' in df.columns:
                        # 基于类别过滤
                        if sentiment_filter == "正面":
                            texts = df[df['sentiment_category'] == 1]['segmented'].dropna()
                        elif sentiment_filter == "负面":
                            texts = df[df['sentiment_category'] == 0]['segmented'].dropna()
                        else:
                            texts = df['segmented'].dropna()
                    else:
                        texts = df['segmented'].dropna()

                    if 'sentiment_score' in df.columns:
                        if sentiment_filter == "正面":
                            texts = df[df['sentiment_score'] > 0.6]['segmented'].dropna()
                        elif sentiment_filter == "负面":
                            texts = df[df['sentiment_score'] < 0.4]['segmented'].dropna()

                    # 准备语料
                    texts = [text for text in texts if isinstance(text, list) and len(text) > 0]

                    if len(texts) < 10:
                        st.error("文本数据不足，至少需要10条")
                    else:
                        # 创建词典和语料库
                        dictionary = corpora.Dictionary(texts)
                        corpus = [dictionary.doc2bow(text) for text in texts]

                        # 训练LDA模型
                        lda_model = models.LdaModel(
                            corpus,
                            num_topics=num_topics,
                            id2word=dictionary,
                            passes=iterations,
                            random_state=42
                        )

                        # 显示主题
                        st.success("✅ 主题建模完成！")

                        for i in range(num_topics):
                            st.write(f"**主题 {i + 1}:**")
                            words = lda_model.print_topic(i, 10)
                            st.write(words)
                            st.write("---")

                except Exception as e:
                    st.error(f"主题建模失败: {str(e)}")

# 聚类分析页面
elif page == "聚类分析":
    st.header("🔍 聚类分析")

    # 显示已有结果
    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists('output/4.2/dbscan_opinion_leaders_improved.png'):
            st.subheader("DBSCAN聚类结果")
            st.image('output/4.2/dbscan_opinion_leaders_improved.png')

    with col2:
        if os.path.exists('output/4.2/kmeans_opinion_leaders.png'):
            st.subheader("K-Means聚类结果")
            st.image('output/4.2/kmeans_opinion_leaders.png')

    # 新建聚类分析
    st.subheader("🆕 新建聚类分析")

    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            cluster_method = st.selectbox("聚类方法", ["K-Means", "DBSCAN"])

            if cluster_method == "K-Means":
                n_clusters = st.slider("聚类数量", 2, 10, 3)
            else:
                eps = st.slider("DBSCAN eps参数", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("DBSCAN min_samples", 2, 20, 5)

        with col2:
            feature_type = st.selectbox("特征类型", ["TF-IDF", "Word2Vec"])
            max_features = st.slider("最大特征数", 100, 2000, 500)

        if st.button("🔍 开始聚类"):
            with st.spinner("正在进行聚类分析..."):
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.cluster import KMeans, DBSCAN
                    from sklearn.decomposition import PCA

                    # 准备文本数据
                    if 'segmented_text' in df.columns:
                        texts = df['segmented_text'].fillna('')
                    elif 'cleaned_text' in df.columns:
                        texts = df['cleaned_text'].fillna('')
                    else:
                        st.error("未找到处理后的文本数据")
                        st.stop()

                    # 特征提取
                    vectorizer = TfidfVectorizer(max_features=max_features)
                    features = vectorizer.fit_transform(texts)

                    # 聚类
                    if cluster_method == "K-Means":
                        model = KMeans(n_clusters=n_clusters, random_state=42)
                    else:
                        model = DBSCAN(eps=eps, min_samples=min_samples)

                    labels = model.fit_predict(features)

                    # PCA降维可视化
                    pca = PCA(n_components=2)
                    features_2d = pca.fit_transform(features.toarray())

                    # 绘制结果
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                                         c=labels, cmap='tab10', alpha=0.6)
                    ax.set_title(f'{cluster_method}聚类结果')
                    ax.set_xlabel('主成分1')
                    ax.set_ylabel('主成分2')
                    plt.colorbar(scatter, ax=ax)
                    st.pyplot(fig)

                    # 聚类统计
                    st.subheader("📊 聚类统计")
                    cluster_counts = pd.Series(labels).value_counts().sort_index()

                    fig, ax = plt.subplots(figsize=(8, 6))
                    cluster_counts.plot(kind='bar', ax=ax)
                    ax.set_title('各聚类大小')
                    ax.set_xlabel('聚类ID')
                    ax.set_ylabel('样本数')
                    st.pyplot(fig)

                    # 显示聚类详情
                    st.subheader("📝 聚类详情")
                    for cluster_id in sorted(cluster_counts.index):
                        if cluster_id != -1:  # 跳过噪声点
                            cluster_data = df[labels == cluster_id]
                            with st.expander(f"聚类 {cluster_id} ({len(cluster_data)} 个样本)"):
                                # 检查可用的情感列
                                if 'sentiment_score' in cluster_data.columns:
                                    st.write(f"平均情感分数: {cluster_data['sentiment_score'].mean():.3f}")
                                elif 'sentiment_category' in cluster_data.columns:
                                    positive_ratio = (cluster_data['sentiment_category'] == 1).mean()
                                    st.write(f"正面比例: {positive_ratio:.2%}")

                                # 显示文本样例
                                text_col = None
                                if 'text' in cluster_data.columns:
                                    text_col = 'text'
                                elif 'cleaned_text' in cluster_data.columns:
                                    text_col = 'cleaned_text'

                                if text_col:
                                    st.write("样例文本:")
                                    samples = cluster_data.sample(min(3, len(cluster_data)))
                                    for _, row in samples.iterrows():
                                        st.write(f"- {row[text_col][:100]}...")

                except Exception as e:
                    st.error(f"聚类分析失败: {str(e)}")

# 社交网络分析页面
elif page == "社交网络分析":
    st.header("🌐 社交网络分析")

    # 显示已有结果
    if os.path.exists('output/4.3/influence_network_analysis.png'):
        st.subheader("📊 影响力网络分析")
        st.image('output/4.3/influence_network_analysis.png')

    if os.path.exists('output/4.3/folium_heatmap.html'):
        st.subheader("🗺️ 地理分布热力图")
        with open('output/4.3/folium_heatmap.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)

    # 用户影响力分析
    if df is not None and 'user_name' in df.columns:
        st.subheader("👥 用户影响力分析")

        # 计算用户统计
        agg_dict = {'comment_id': 'count'}

        if 'like_count' in df.columns:
            agg_dict['like_count'] = 'sum'
        if 'reply_count' in df.columns:
            agg_dict['reply_count'] = 'sum'

        # 检查情感分数列
        sentiment_col = None
        if 'sentiment_score' in df.columns:
            sentiment_col = 'sentiment_score'
            agg_dict['sentiment_score'] = 'mean'
        elif 'sentiment_category' in df.columns:
            # 计算正面比例作为情感指标
            df['positive_flag'] = (df['sentiment_category'] == 1).astype(int)
            agg_dict['positive_flag'] = 'mean'

        user_stats = df.groupby('user_name').agg(agg_dict)

        # 重命名列
        user_stats = user_stats.rename(columns={'comment_id': 'comment_count'})

        # 计算影响力分数
        if 'like_count' in user_stats.columns and 'reply_count' in user_stats.columns:
            user_stats['influence_score'] = (
                    user_stats['like_count'] * 0.5 +
                    user_stats['reply_count'] * 0.3 +
                    user_stats['comment_count'] * 0.2
            )
        else:
            # 简单基于评论数的影响力
            user_stats['influence_score'] = user_stats['comment_count']

        # Top用户
        top_users = user_stats.nlargest(10, 'influence_score')

        # 显示图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 影响力排名
        top_users['influence_score'].plot(kind='barh', ax=ax1)
        ax1.set_title('Top 10 影响力用户')
        ax1.set_xlabel('影响力分数')

        # 情感vs影响力
        if sentiment_col in user_stats.columns:
            ax2.scatter(user_stats['influence_score'], user_stats[sentiment_col], alpha=0.6)
            ax2.set_xlabel('影响力分数')
            ax2.set_ylabel('平均情感分数')
            ax2.set_title('用户影响力与情感倾向')
            ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        elif 'positive_flag' in user_stats.columns:
            ax2.scatter(user_stats['influence_score'], user_stats['positive_flag'], alpha=0.6)
            ax2.set_xlabel('影响力分数')
            ax2.set_ylabel('正面评论比例')
            ax2.set_title('用户影响力与正面倾向')
            ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

        st.pyplot(fig)

        # 显示详细信息
        st.subheader("📋 影响力用户详情")
        st.dataframe(top_users.round(2))

# 综合报告页面
elif page == "综合报告":
    st.header("📑 综合分析报告")

    if df is not None:
        # 生成报告
        report = "# 微博饭圈文化舆情分析系统综合报告\n\n"
        report += f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # 数据概览
        report += "## 1. 数据概览\n\n"
        report += f"- 总评论数: {len(df):,} 条\n"

        if 'user_name' in df.columns:
            report += f"- 活跃用户数: {df['user_name'].nunique():,} 人\n"

        if 'datetime' in df.columns:
            date_range = f"{df['datetime'].min()} 至 {df['datetime'].max()}"
            report += f"- 时间范围: {date_range}\n"

        # 情感分析
        sentiment_col = None
        if 'sentiment_score' in df.columns:
            sentiment_col = 'sentiment_score'
        elif 'sentiment_category' in df.columns:
            # 基于类别生成简单统计
            report += "\n## 2. 情感分析\n\n"
            if 'sentiment_category' in df.columns:
                sentiment_counts = df['sentiment_category'].value_counts()
                if 1 in sentiment_counts:
                    report += f"- 正面评论: {sentiment_counts.get(1, 0)} 条\n"
                if 0 in sentiment_counts:
                    report += f"- 负面评论: {sentiment_counts.get(0, 0)} 条\n"
                if 2 in sentiment_counts:
                    report += f"- 中性评论: {sentiment_counts.get(2, 0)} 条\n"

        if sentiment_col:
            report += "\n## 2. 情感分析\n\n"
            positive_ratio = (df[sentiment_col] > 0.6).mean()
            neutral_ratio = ((df[sentiment_col] >= 0.4) & (df[sentiment_col] <= 0.6)).mean()
            negative_ratio = (df[sentiment_col] < 0.4).mean()

            report += f"- 正面评论: {positive_ratio:.1%}\n"
            report += f"- 中性评论: {neutral_ratio:.1%}\n"
            report += f"- 负面评论: {negative_ratio:.1%}\n"
            report += f"- 平均情感分数: {df[sentiment_col].mean():.3f}\n"

        # 热门话题
        if 'segmented' in df.columns:
            report += "\n## 3. 热门话题\n\n"
            all_words = []
            for words in df['segmented']:
                if isinstance(words, list):
                    all_words.extend(words)

            if all_words:
                word_freq = pd.Series(all_words).value_counts().head(10)
                for word, count in word_freq.items():
                    report += f"- {word}: {count} 次\n"

        # 结论与建议
        report += "\n## 4. 结论与建议\n\n"

        if sentiment_col:
            avg_sentiment = df[sentiment_col].mean()
            if avg_sentiment > 0.6:
                report += "- 整体舆情偏正面，品牌形象良好\n"
                report += "- 建议继续保持现有策略，加强正面内容传播\n"
            elif avg_sentiment < 0.4:
                report += "- 整体舆情偏负面，需要重点关注\n"
                report += "- 建议及时回应负面反馈，改善产品或服务\n"
            else:
                report += "- 整体舆情较为中性\n"
                report += "- 建议加强互动，提升用户参与度\n"

        # 显示报告
        st.markdown(report)

        # 下载按钮
        st.download_button(
            label="📥 下载报告",
            data=report,
            file_name=f"舆情分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

        # 显示所有图表
        st.subheader("📊 分析图表汇总")

        # 展示output目录下的所有图片
        output_dirs = ['output/2', 'output/3.3', 'output/4.1', 'output/4.2', 'output/4.3']

        for dir_path in output_dirs:
            if os.path.exists(dir_path):
                st.write(f"**{dir_path} 目录下的图表:**")

                image_files = [f for f in os.listdir(dir_path)
                               if f.endswith(('.png', '.jpg', '.jpeg'))]

                if image_files:
                    cols = st.columns(2)
                    for i, img_file in enumerate(image_files):
                        with cols[i % 2]:
                            img_path = os.path.join(dir_path, img_file)
                            st.image(img_path, caption=img_file, use_column_width=True)
                else:
                    st.write("- 无图表文件")

                st.write("---")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>微博饭圈文化舆情分析系统 | 基于Streamlit开发（22级数据科学与大数据技术1班 202205050107 李元鑫）</p>
    </div>
    """,
    unsafe_allow_html=True
)