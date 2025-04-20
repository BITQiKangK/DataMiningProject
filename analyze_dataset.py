import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import sys
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset(dataset_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取所有Parquet文件
    print(f"Reading data files from {dataset_path}...")
    parquet_files = glob.glob(f'{dataset_path}/*.parquet')
    if not parquet_files:
        print(f"No parquet files found in {dataset_path}")
        return
        
    data_frames = []
    
    for file in tqdm(parquet_files, desc="Reading files"):
        df = pd.read_parquet(file, engine='pyarrow')
        data_frames.append(df)
    
    # 合并所有数据
    df = pd.concat(data_frames, ignore_index=True)
    print(f"Original dataset shape: {df.shape}")
    
    # 数据预处理
    print("\nPerforming data preprocessing...")
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values Count': missing_values,
        'Missing Percentage': missing_percentage.round(2)
    })
    
    print("\nMissing values information:")
    print(missing_info[missing_info['Missing Values Count'] > 0])
    
    # 处理可能的异常值
    print("\nDetecting and handling outliers...")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 使用箱线图方法检测异常值
    outliers_count = {}
    for col in numeric_columns:
        if df[col].isnull().sum() / len(df) < 0.3:  # 仅处理缺失值比例小于30%的列
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            outliers_count[col] = outliers
            
            # 可视化异常值
            if outliers > 0:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[col])
                plt.title(f'Boxplot of {col} - Detected {outliers} outliers')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{col}_outliers.png')
                plt.close()
    
    print("\nOutliers statistics:")
    for col, count in outliers_count.items():
        if count > 0:
            print(f"{col}: {count} outliers ({count/len(df)*100:.2f}%)")
    
    # 识别潜在高价值用户
    print("\nStarting to identify high-value users...")
    
    # 显示数据集的列
    print("\nDataset columns:")
    print(df.columns.tolist())
    
    try:
        # 方法: K-均值聚类
        print("\nPerforming K-means clustering analysis...")
        
        # 选择数值型特征进行聚类
        clustering_features = [col for col in numeric_columns if df[col].isnull().sum() / len(df) < 0.1]
        
        if len(clustering_features) >= 2:
            # 填充缺失值
            df_clustering = df[clustering_features].copy()
            for col in clustering_features:
                df_clustering[col] = df_clustering[col].fillna(df_clustering[col].mean())
            
            # 标准化数据
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_clustering)
            
            # 确定最佳聚类数
            inertia = []
            K_range = range(2, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_features)
                inertia.append(kmeans.inertia_)
            
            # 绘制肘部图
            plt.figure(figsize=(10, 6))
            plt.plot(K_range, inertia, 'bo-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal Cluster Number')
            plt.savefig(f'{output_dir}/elbow_method.png')
            plt.close()
            
            # 根据肘部图确定聚类数，这里我们假设是3
            # 实际应用中，应该根据肘部图来确定
            n_clusters = 3  # 可以根据肘部图结果调整
            
            # 执行K-均值聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # 将聚类结果添加到数据框
            df_clustering['cluster'] = cluster_labels
            
            # 分析各个聚类的特征
            cluster_analysis = df_clustering.groupby('cluster').mean()
            print("\nFeature values of cluster centers:")
            print(cluster_analysis)
            
            # 保存聚类中心信息
            cluster_analysis.to_csv(f'{output_dir}/cluster_centers.csv')
            
            # 确定哪个聚类代表高价值用户
            # 假设特征值越高，用户价值越高
            # 我们计算每个聚类的特征均值之和
            cluster_analysis['feature_sum'] = cluster_analysis.sum(axis=1)
            high_value_cluster = cluster_analysis['feature_sum'].idxmax()
            
            print(f"\nCluster {high_value_cluster} likely represents high-value user group")
            
            # 统计高价值用户数量
            high_value_count = sum(cluster_labels == high_value_cluster)
            high_value_percentage = high_value_count/len(df)*100
            print(f"Number of high-value users: {high_value_count} ({high_value_percentage:.2f}%)")
            
            # 可视化聚类结果（使用PCA降维到2D）
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(scaled_features)
            
            # 创建包含降维结果和聚类标签的DataFrame
            df_pca = pd.DataFrame(reduced_features, columns=['Component 1', 'Component 2'])
            df_pca['Cluster'] = cluster_labels
            
            # 标记高价值聚类
            df_pca['User Type'] = df_pca['Cluster'].apply(lambda x: 'High-Value Users' if x == high_value_cluster else 'Regular Users')
            
            # 使用Plotly绘制交互式散点图
            fig = px.scatter(
                df_pca, 
                x='Component 1', 
                y='Component 2', 
                color='User Type',
                color_discrete_map={'High-Value Users': 'red', 'Regular Users': 'blue'},
                title='User Clustering Analysis (PCA Dimension Reduction)',
                hover_data=['Cluster']
            )
            fig.write_html(f'{output_dir}/user_clustering.html')
            
            # 保存高价值用户信息
            if 'user_id' in df.columns:
                user_id_col = 'user_id'
            else:
                id_cols = [col for col in df.columns if 'id' in col.lower()]
                user_id_col = id_cols[0] if id_cols else df.index.name or 'index'
                
            if user_id_col in df.columns:
                # 创建包含用户ID和聚类标签的DataFrame
                user_clusters = pd.DataFrame({
                    'user_id': df[user_id_col],
                    'cluster': cluster_labels,
                    'is_high_value': cluster_labels == high_value_cluster
                })
                high_value_users = user_clusters[user_clusters['is_high_value']]
                high_value_users.to_csv(f'{output_dir}/high_value_users_clustering.csv', index=False)
            
            # 生成中英文分析报告摘要
            # 英文报告
            with open(f'{output_dir}/analysis_summary_en.md', 'w', encoding='utf-8') as f:
                f.write("# High-Value User Analysis Report\n\n")
                f.write("## Analysis Methods\n\n")
                f.write("This analysis used the following methods to identify potential high-value users:\n\n")
                f.write("1. **K-means Clustering**: Using numerical features to cluster users with similar behavior patterns\n\n")
                
                f.write("## Data Overview\n\n")
                f.write(f"* Dataset size: {df.shape[0]} records with {df.shape[1]} features\n")
                f.write(f"* Features include: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}\n")
                
                if missing_info[missing_info['Missing Values Count'] > 0].empty:
                    f.write("* No missing values were found in the dataset\n\n")
                else:
                    f.write("* Missing values were found in the dataset\n\n")
                
                f.write("## Key Findings\n\n")
                
                f.write("### Cluster Analysis Results\n\n")
                f.write(f"{n_clusters} distinct user clusters were identified:\n\n")
                
                # 为每个聚类写入特征信息
                for i in range(n_clusters):
                    features_text = []
                    for col in clustering_features:
                        if "age" in col.lower():
                            features_text.append(f"average age {cluster_analysis.loc[i, col]:.1f}")
                        elif "income" in col.lower():
                            features_text.append(f"income ${cluster_analysis.loc[i, col]:.0f}")
                        elif "credit" in col.lower() or "score" in col.lower():
                            features_text.append(f"credit score {cluster_analysis.loc[i, col]:.0f}")
                        else:
                            features_text.append(f"{col} {cluster_analysis.loc[i, col]:.2f}")
                    
                    f.write(f"* **Cluster {i}**: {', '.join(features_text)}\n")
                
                f.write(f"\n**Cluster {high_value_cluster}** was identified as the high-value user group, representing {high_value_percentage:.2f}% of all users.\n\n")
                
                if 'age' in df_clustering.columns and 'income' in df_clustering.columns and 'credit_score' in df_clustering.columns:
                    avg_age = cluster_analysis.loc[high_value_cluster, 'age']
                    avg_income = cluster_analysis.loc[high_value_cluster, 'income']
                    avg_credit = cluster_analysis.loc[high_value_cluster, 'credit_score']
                    
                    f.write("## Recommendations\n\n")
                    f.write(f"* **Targeted Marketing**: Implement personalized marketing strategies for the identified high-value users in Cluster {high_value_cluster}\n")
                    f.write(f"* **Age-Focused Approach**: Since high-value users are {'middle-aged' if 35 <= avg_age <= 55 else 'young' if avg_age < 35 else 'senior'} (around {avg_age:.1f} years), design age-appropriate products and services\n")
                    
                    if avg_credit < 600:
                        f.write(f"* **Credit Building**: Consider offering credit-building products to high-value users as they have moderate credit scores ({avg_credit:.0f}) but high income\n")
                    
                    f.write("* **Loyalty Programs**: Develop customer loyalty programs to further increase retention rates of high-value users\n")
            
            # 中文报告
            with open(f'{output_dir}/分析报告.md', 'w', encoding='utf-8') as f:
                f.write("# 高价值用户分析报告\n\n")
                f.write("## 分析方法\n\n")
                f.write("本分析使用了以下方法来识别潜在的高价值用户：\n\n")
                f.write("1. **K-均值聚类**：使用用户的数值特征进行聚类，识别具有相似行为模式的用户群体\n\n")
                
                f.write("## 数据概述\n\n")
                f.write(f"* 数据集大小：{df.shape[0]} 条记录，{df.shape[1]} 个特征\n")
                f.write(f"* 特征包括：{', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}\n")
                
                if missing_info[missing_info['Missing Values Count'] > 0].empty:
                    f.write("* 数据集中未发现缺失值\n\n")
                else:
                    f.write("* 数据集中存在缺失值\n\n")
                
                f.write("## 主要发现\n\n")
                
                f.write("### 聚类分析结果\n\n")
                f.write(f"通过 K-均值聚类分析，我们识别出{n_clusters}个明显不同的用户群体：\n\n")
                
                # 为每个聚类写入特征信息
                for i in range(n_clusters):
                    features_text = []
                    for col in clustering_features:
                        if "age" in col.lower():
                            features_text.append(f"平均年龄 {cluster_analysis.loc[i, col]:.1f}岁")
                        elif "income" in col.lower():
                            features_text.append(f"收入 {cluster_analysis.loc[i, col]:.0f}美元")
                        elif "credit" in col.lower() or "score" in col.lower():
                            features_text.append(f"信用评分 {cluster_analysis.loc[i, col]:.0f}")
                        else:
                            features_text.append(f"{col} {cluster_analysis.loc[i, col]:.2f}")
                    
                    f.write(f"* **聚类 {i}**：{', '.join(features_text)}\n")
                
                f.write(f"\n**聚类 {high_value_cluster}** 被确定为高价值用户群体，占所有用户的 {high_value_percentage:.2f}%。\n\n")
                
                if 'age' in df_clustering.columns and 'income' in df_clustering.columns and 'credit_score' in df_clustering.columns:
                    avg_age = cluster_analysis.loc[high_value_cluster, 'age']
                    avg_income = cluster_analysis.loc[high_value_cluster, 'income']
                    avg_credit = cluster_analysis.loc[high_value_cluster, 'credit_score']
                    
                    f.write("## 建议\n\n")
                    f.write(f"* **精准营销**：对聚类 {high_value_cluster} 中的高价值用户实施个性化的营销策略\n")
                    
                    age_description = "中年" if 35 <= avg_age <= 55 else "年轻" if avg_age < 35 else "年长"
                    f.write(f"* **年龄导向方法**：由于高价值用户平均年龄在 {avg_age:.1f} 岁左右（{age_description}人群），设计适合这个年龄段的产品和服务\n")
                    
                    if avg_credit < 600:
                        f.write(f"* **信用提升计划**：考虑向高价值用户提供信用建设产品，因为他们具有中等信用评分（{avg_credit:.0f}）但收入较高\n")
                    
                    f.write("* **忠诚度计划**：开发客户忠诚度计划，进一步提高高价值用户的留存率\n")
                
                f.write("\n## 业务价值\n\n")
                f.write(f"基于我们的分析，专注于聚类 {high_value_cluster} 中的高价值用户群体可以带来以下业务价值：\n\n")
                
                if 'income' in df_clustering.columns:
                    f.write(f"1. **收入增长**：这部分用户收入高达{avg_income:.0f}美元，具有强大的消费能力\n")
                
                if 'credit_score' in df_clustering.columns and avg_credit < 600:
                    f.write(f"2. **潜在发展空间**：中等信用评分（{avg_credit:.0f}）表明这些用户还有信用提升空间，可以开发相关金融产品\n")
                
                if 'age' in df_clustering.columns and 35 <= avg_age <= 55:
                    f.write(f"3. **稳定客户群**：中年（{avg_age:.1f}岁）客户群体通常有稳定的职业和生活方式，是长期业务关系的理想目标\n")
                
                f.write(f"4. **规模效应**：占总体用户{high_value_percentage:.2f}%的比例意味着针对性策略可以覆盖相当大的客户基础\n")
        else:
            print("Not enough numerical features for clustering")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
    
    print(f"\nHigh-value user analysis completed. Results saved in '{output_dir}' directory.")
    
    return df.shape[0], high_value_percentage if 'high_value_percentage' in locals() else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze dataset and identify high-value users')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    output_dir = args.output_dir or f"high_value_analysis_{os.path.basename(dataset_path)}"
    
    analyze_dataset(dataset_path, output_dir) 