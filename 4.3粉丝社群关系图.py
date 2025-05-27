import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from collections import defaultdict, Counter
import warnings
import os

# Folium imports
import folium
from folium.plugins import HeatMap, MarkerCluster

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('output/4.3', exist_ok=True)


class FanCommunityController:
    """粉丝社群关系分析与行为引导系统"""

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.user_features = {}
        self.similarity_matrix = None
        self.community_graph = None
        self.influence_network = {}
        self.control_strategies = {}

        # 省份坐标（用于地理可视化）
        self.province_coords = {
            '北京': [39.9042, 116.4074], '上海': [31.2304, 121.4737],
            '广东': [23.1291, 113.2644], '江苏': [32.0617, 118.7778],
            '浙江': [30.2741, 120.1551], '山东': [36.6758, 117.0264],
            '河南': [34.7466, 113.6254], '四川': [30.6171, 104.0668],
            '湖北': [30.5928, 114.3058], '湖南': [28.2282, 112.9388],
            '福建': [26.0789, 119.2965], '安徽': [31.8612, 117.2272],
            '江西': [28.6760, 115.9092], '河北': [38.0428, 114.5149],
            '山西': [37.8570, 112.5489], '辽宁': [41.8057, 123.4315],
            '天津': [39.1042, 117.2000], '重庆': [29.5647, 106.5507]
        }

    def preprocess_data(self, max_users=150):
        """数据预处理：识别关键用户和构建特征"""
        user_stats = self.data.groupby('user_name').agg({
            'like_count': 'sum',
            'reply_count': 'sum',
            'comment_id': 'count',
            'sentiment_score': ['mean', 'std'],
            'location': lambda x: x.mode().iloc[0] if len(x.dropna()) > 0 else '未知'
        }).reset_index()

        user_stats.columns = ['user_name', 'total_likes', 'total_replies', 'comment_count',
                              'avg_sentiment', 'sentiment_volatility', 'location']

        user_stats['influence_score'] = (
                user_stats['total_likes'] * 0.3 +
                user_stats['total_replies'] * 0.5 +
                user_stats['comment_count'] * 0.2
        )

        top_users = user_stats.nlargest(max_users, 'influence_score')['user_name'].tolist()
        self.filtered_data = self.data[self.data['user_name'].isin(top_users)].copy()

        for _, user_stat in user_stats.iterrows():
            if user_stat['user_name'] in top_users:
                user_data = self.filtered_data[self.filtered_data['user_name'] == user_stat['user_name']]

                self.user_features[user_stat['user_name']] = {
                    'avg_sentiment': user_stat['avg_sentiment'],
                    'sentiment_volatility': user_stat['sentiment_volatility'] if not pd.isna(
                        user_stat['sentiment_volatility']) else 0,
                    'total_likes': user_stat['total_likes'],
                    'total_replies': user_stat['total_replies'],
                    'comment_count': user_stat['comment_count'],
                    'location': user_stat['location'],
                    'province': self._extract_province(user_stat['location']),
                    'influence_score': user_stat['influence_score'],
                    'influence_type': self._classify_influence_type(user_stat),
                    'control_priority': self._assess_control_priority(user_data),
                    'guidance_strategy': self._recommend_guidance_strategy(user_data, user_stat)
                }

        return self.user_features

    def _extract_province(self, location):
        """提取省份信息"""
        if not location or pd.isna(location):
            return '其他'

        province_keywords = {
            '北京': ['北京'], '上海': ['上海'], '广东': ['广东', '广州', '深圳'],
            '江苏': ['江苏', '南京', '苏州'], '浙江': ['浙江', '杭州', '宁波'],
            '山东': ['山东', '济南', '青岛'], '河南': ['河南', '郑州'],
            '四川': ['四川', '成都'], '湖北': ['湖北', '武汉'],
            '湖南': ['湖南', '长沙'], '福建': ['福建', '福州', '厦门'],
            '安徽': ['安徽', '合肥'], '江西': ['江西', '南昌'],
            '河北': ['河北', '石家庄'], '山西': ['山西', '太原'],
            '辽宁': ['辽宁', '沈阳', '大连'], '天津': ['天津'], '重庆': ['重庆']
        }

        for province, keywords in province_keywords.items():
            if any(keyword in str(location) for keyword in keywords):
                return province
        return '其他'

    def _classify_influence_type(self, user_stat):
        """分类用户影响力类型"""
        likes_ratio = user_stat['total_likes'] / max(1, user_stat['total_likes'] + user_stat['total_replies'])

        if user_stat['avg_sentiment'] > 0.2:
            return "正面引导型" if likes_ratio > 0.6 else "积极互动型"
        elif user_stat['avg_sentiment'] < -0.2:
            return "负面扩散型" if likes_ratio > 0.6 else "争议煽动型"
        else:
            return "中性观察型" if user_stat['comment_count'] > 10 else "低活跃型"

    def _assess_control_priority(self, user_data):
        """评估控制优先级"""
        risk_score = 0

        negative_ratio = (user_data['sentiment_score'] < -0.3).sum() / len(user_data)
        risk_score += negative_ratio * 50

        total_engagement = user_data['like_count'].sum() + user_data['reply_count'].sum()
        if total_engagement > 500 and user_data['sentiment_score'].mean() < -0.1:
            risk_score += 30

        sentiment_std = user_data['sentiment_score'].std()
        if not pd.isna(sentiment_std) and sentiment_std > 0.5:
            risk_score += 20

        if risk_score > 70:
            return "高优先级"
        elif risk_score > 40:
            return "中优先级"
        else:
            return "低优先级"

    def _recommend_guidance_strategy(self, user_data, user_stat):
        """推荐引导策略"""
        influence_score = user_stat['influence_score']
        avg_sentiment = user_stat['avg_sentiment']

        if avg_sentiment > 0.2 and influence_score > 100:
            return "合作推广"
        elif avg_sentiment < -0.2 and influence_score > 100:
            return "重点监控"
        elif avg_sentiment < -0.1:
            return "情感转化"
        elif -0.1 <= avg_sentiment <= 0.1:
            return "中性维护"
        else:
            return "正面强化"

    def build_similarity_network(self):
        """基于评论相似度构建社群网络"""
        user_texts = self.filtered_data.groupby('user_name')['cleaned_text'].apply(
            lambda x: ' '.join(x)
        ).reset_index()

        vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=2,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(user_texts['cleaned_text'])

        self.similarity_matrix = cosine_similarity(tfidf_matrix)

        self.community_graph = nx.Graph()

        for i, user in enumerate(user_texts['user_name']):
            if user in self.user_features:
                features = self.user_features[user]
                self.community_graph.add_node(user, **features)

        threshold = 0.3
        for i in range(len(self.similarity_matrix)):
            for j in range(i + 1, len(self.similarity_matrix)):
                if self.similarity_matrix[i][j] > threshold:
                    user1 = user_texts['user_name'].iloc[i]
                    user2 = user_texts['user_name'].iloc[j]
                    if user1 in self.user_features and user2 in self.user_features:
                        similarity = self.similarity_matrix[i][j]
                        self.community_graph.add_edge(user1, user2,
                                                      weight=similarity,
                                                      influence_flow=self._calculate_influence_flow(user1, user2))

        return self.community_graph

    def _calculate_influence_flow(self, user1, user2):
        """计算用户间影响力流动方向"""
        influence1 = self.user_features[user1]['influence_score']
        influence2 = self.user_features[user2]['influence_score']

        if influence1 > influence2 * 1.5:
            return f"{user1} → {user2}"
        elif influence2 > influence1 * 1.5:
            return f"{user2} → {user1}"
        else:
            return f"{user1} ↔ {user2}"

    def detect_communities_and_leaders(self):
        """检测社群并识别关键节点"""
        if self.community_graph.number_of_nodes() == 0:
            return {}

        communities = nx.algorithms.community.greedy_modularity_communities(self.community_graph)

        community_analysis = {}
        for i, community in enumerate(communities):
            if len(community) >= 3:
                valid_community_users = [user for user in community if user in self.user_features]

                if len(valid_community_users) >= 3:
                    sentiments = [self.user_features[user]['avg_sentiment'] for user in valid_community_users]
                    influences = [self.user_features[user]['influence_score'] for user in valid_community_users]
                    provinces = [self.user_features[user]['province'] for user in valid_community_users]

                    leader = max(valid_community_users, key=lambda u: self.user_features[u]['influence_score'])

                    risk_users = [user for user in valid_community_users
                                  if self.user_features[user]['control_priority'] in ['高优先级', '中优先级']]

                    community_analysis[f'社群{i + 1}'] = {
                        'size': len(valid_community_users),
                        'members': valid_community_users,
                        'leader': leader,
                        'avg_sentiment': np.mean(sentiments),
                        'total_influence': sum(influences),
                        'dominant_provinces': Counter(provinces).most_common(3),
                        'risk_users': risk_users,
                        'control_strategy': self._design_community_control_strategy(valid_community_users),
                        'guidance_approach': self._design_guidance_approach(sentiments, influences)
                    }

        self.communities = community_analysis
        return community_analysis

    def _design_community_control_strategy(self, community_users):
        """为社群设计控制策略"""
        priorities = [self.user_features[user]['control_priority'] for user in community_users]
        high_priority_count = priorities.count('高优先级')

        if high_priority_count > len(community_users) * 0.3:
            return "重点监控社群"
        elif high_priority_count > 0:
            return "定向干预社群"
        else:
            return "维护引导社群"

    def _design_guidance_approach(self, sentiments, influences):
        """设计引导方法"""
        avg_sentiment = np.mean(sentiments)
        max_influence = max(influences)

        if avg_sentiment < -0.2 and max_influence > 200:
            return "领袖转化策略"
        elif avg_sentiment < -0.1:
            return "正面内容注入"
        elif avg_sentiment > 0.1:
            return "正面强化维护"
        else:
            return "中性平衡维护"

    def create_influence_network_visualization(self):
        """创建影响力网络可视化"""
        import matplotlib
        matplotlib.use('Agg')

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        if self.community_graph and self.community_graph.number_of_nodes() > 0:
            pos = nx.spring_layout(self.community_graph, k=3, iterations=50)

            node_colors = []
            node_sizes = []
            valid_nodes = []

            for node in self.community_graph.nodes():
                if node in self.user_features:
                    valid_nodes.append(node)
                    priority = self.user_features[node]['control_priority']
                    influence = self.user_features[node]['influence_score']

                    if priority == '高优先级':
                        node_colors.append('#FF0000')
                    elif priority == '中优先级':
                        node_colors.append('#FFA500')
                    else:
                        node_colors.append('#90EE90')

                    node_sizes.append(max(50, min(500, influence * 2)))

            if valid_nodes:
                valid_subgraph = self.community_graph.subgraph(valid_nodes)
                valid_pos = {node: pos[node] for node in valid_nodes if node in pos}

                if valid_pos:
                    nx.draw_networkx_nodes(valid_subgraph, valid_pos,
                                           node_color=node_colors, node_size=node_sizes,
                                           alpha=0.8, ax=ax1)
                    nx.draw_networkx_edges(valid_subgraph, valid_pos,
                                           alpha=0.3, width=0.5, ax=ax1)

                    high_priority_users = [user for user in valid_nodes
                                           if self.user_features[user]['control_priority'] == '高优先级'][:5]
                    high_priority_pos = {user: valid_pos[user] for user in high_priority_users
                                         if user in valid_pos}

                    if high_priority_pos:
                        labels = {user: user[:4] + '...' for user in high_priority_pos.keys()}
                        nx.draw_networkx_labels(valid_subgraph, high_priority_pos,
                                                labels=labels, font_size=20, ax=ax1)

            ax1.set_title('用户影响力网络\n(红色=高优先级, 橙色=中优先级, 绿色=低优先级)',
                          fontsize=24, fontweight='bold')
            ax1.axis('off')

        province_strategy = defaultdict(lambda: {'高优先级': 0, '中优先级': 0, '低优先级': 0})
        for user, features in self.user_features.items():
            province = features['province']
            priority = features['control_priority']
            if province != '其他':
                province_strategy[province][priority] += 1

        if province_strategy:
            provinces = list(province_strategy.keys())
            high_counts = [province_strategy[p]['高优先级'] for p in provinces]
            medium_counts = [province_strategy[p]['中优先级'] for p in provinces]
            low_counts = [province_strategy[p]['低优先级'] for p in provinces]

            x = np.arange(len(provinces))
            width = 0.6

            ax2.bar(x, high_counts, width, label='高优先级', color='#FF6B6B')
            ax2.bar(x, medium_counts, width, bottom=high_counts, label='中优先级', color='#FFA726')
            ax2.bar(x, low_counts, width,
                    bottom=np.array(high_counts) + np.array(medium_counts),
                    label='低优先级', color='#66BB6A')

            ax2.set_xlabel('省份', fontsize=24)
            ax2.set_ylabel('用户数量', fontsize=24)
            ax2.set_title('各省份用户控制优先级分布', fontweight='bold', fontsize=24)
            ax2.set_xticks(x)
            ax2.set_xticklabels(provinces, rotation=45, fontsize=24)
            ax2.legend(fontsize=24)

        influence_types = [features['influence_type'] for features in self.user_features.values()]
        type_counts = Counter(influence_types)

        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

            wedges, texts, autotexts = ax3.pie(counts, labels=types, autopct='%1.1f%%',
                                               colors=colors[:len(types)], textprops={'fontsize': 24})
            ax3.set_title('用户影响力类型分布', fontweight='bold', fontsize=24)

        strategies = [features['guidance_strategy'] for features in self.user_features.values()]
        strategy_counts = Counter(strategies)

        if strategy_counts:
            strategy_names = list(strategy_counts.keys())
            strategy_values = list(strategy_counts.values())

            bars = ax4.barh(strategy_names, strategy_values, color='skyblue')
            ax4.set_xlabel('用户数量', fontsize=24)
            ax4.set_title('推荐引导策略分布', fontweight='bold', fontsize=24)
            ax4.tick_params(axis='y', labelsize=24)
            ax4.tick_params(axis='x', labelsize=24)

            for bar in bars:
                width = bar.get_width()
                ax4.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{int(width)}', ha='left', va='center', fontsize=24)

        plt.tight_layout()
        plt.savefig('./output/4.3/influence_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def create_geographic_heatmap(self):
        """创建地理热力图"""
        province_stats = defaultdict(lambda: {
            'user_count': 0, 'high_priority_count': 0, 'avg_sentiment': [],
            'total_influence': 0, 'control_strategies': []
        })

        for user, features in self.user_features.items():
            province = features['province']
            if province in self.province_coords:
                stats = province_stats[province]
                stats['user_count'] += 1
                stats['avg_sentiment'].append(features['avg_sentiment'])
                stats['total_influence'] += features['influence_score']
                stats['control_strategies'].append(features['guidance_strategy'])

                if features['control_priority'] == '高优先级':
                    stats['high_priority_count'] += 1

        m = folium.Map(location=[35.8617, 104.1954], zoom_start=4, tiles='OpenStreetMap')

        heat_data = []

        for province, stats in province_stats.items():
            if stats['user_count'] > 0:
                lat, lon = self.province_coords[province]

                avg_sentiment = np.mean(stats['avg_sentiment'])
                avg_influence = stats['total_influence'] / stats['user_count']
                risk_ratio = stats['high_priority_count'] / stats['user_count']

                heat_data.append([lat, lon, risk_ratio * 100])

                if risk_ratio > 0.3:
                    color = 'red'
                elif risk_ratio > 0.1:
                    color = 'orange'
                else:
                    color = 'green'

                most_common_strategy = Counter(stats['control_strategies']).most_common(1)[0][0]

                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h3 style="margin-bottom: 10px; color: #333; text-align: center;">{province}省粉丝分析</h3>
                    <hr>
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <tr><td><b>用户总数:</b></td><td>{stats['user_count']}人</td></tr>
                        <tr><td><b>高优先级用户:</b></td><td>{stats['high_priority_count']}人 ({risk_ratio:.1%})</td></tr>
                        <tr><td><b>平均情感指数:</b></td><td>{avg_sentiment:+.3f}</td></tr>
                        <tr><td><b>平均影响力:</b></td><td>{avg_influence:.1f}</td></tr>
                        <tr><td><b>推荐策略:</b></td><td>{most_common_strategy}</td></tr>
                    </table>
                    <hr>
                    <p style="margin: 5px 0; font-size: 11px; color: #666;">
                        <b>控制建议:</b> {'重点关注' if risk_ratio > 0.3 else '适度监控' if risk_ratio > 0.1 else '维持现状'}
                    </p>
                </div>
                """

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=min(30, max(8, stats['user_count'] * 1.5)),
                    popup=folium.Popup(popup_html, max_width=320),
                    color=color,
                    fill=True,
                    fillOpacity=0.7,
                    weight=3
                ).add_to(m)

                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=f'<div style="color: black; font-weight: bold; font-size: 11px; text-shadow: 1px 1px 1px white;">{province}</div>',
                        icon_size=(50, 20),
                        icon_anchor=(25, 10)
                    )
                ).add_to(m)

        if heat_data:
            HeatMap(heat_data, radius=40, blur=25, max_zoom=1).add_to(m)

        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: 160px; 
                    background-color: white; border: 2px solid grey; z-index: 9999; 
                    font-size: 13px; padding: 15px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
        <h4 style="margin-top: 0; color: #333;">粉丝控制策略图例</h4>
        <p><span style="color: red; font-size: 16px;">●</span> 高风险区域 - 重点监控</p>
        <p><span style="color: orange; font-size: 16px;">●</span> 中风险区域 - 适度关注</p>
        <p><span style="color: green; font-size: 16px;">●</span> 低风险区域 - 维持现状</p>
        <p style="font-size: 11px; color: #666; margin-top: 10px;">
        圆圈大小 = 用户数量<br>
        热力图强度 = 风险程度
        </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        m.save('./output/4.3/folium_heatmap.html')

    def generate_control_strategy_report(self):
        """生成控制策略报告"""
        priorities = [f['control_priority'] for f in self.user_features.values()]
        priority_counts = Counter(priorities)

        influence_types = [f['influence_type'] for f in self.user_features.values()]
        type_counts = Counter(influence_types)

        province_risks = defaultdict(int)
        for features in self.user_features.values():
            if features['province'] != '其他' and features['control_priority'] == '高优先级':
                province_risks[features['province']] += 1

        high_priority_users = [user for user, f in self.user_features.items()
                               if f['control_priority'] == '高优先级']

        positive_leaders = [user for user, f in self.user_features.items()
                            if f['avg_sentiment'] > 0.2 and f['influence_score'] > 100]

        return {
            'total_users': len(self.user_features),
            'high_priority_users': len(high_priority_users),
            'positive_leaders': len(positive_leaders),
            'risk_provinces': list(province_risks.keys()),
            'communities_count': len(getattr(self, 'communities', {}))
        }

    def run_complete_analysis(self):
        """运行完整的控制导向分析"""
        self.preprocess_data()
        self.build_similarity_network()
        self.detect_communities_and_leaders()
        self.create_influence_network_visualization()
        self.create_geographic_heatmap()
        report = self.generate_control_strategy_report()
        return report


if __name__ == "__main__":
    data_file = './data/sentiment_processed_data.csv'
    controller = FanCommunityController(data_file)
    report = controller.run_complete_analysis()