import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取所有Parquet文件
print("正在读取数据文件...")
parquet_files = glob.glob('dataset/1G_data/*.parquet')
data_frames = []

for file in tqdm(parquet_files, desc="读取文件"):
    # 只读取部分数据用于初步探索
    df = pd.read_parquet(file, engine='pyarrow')
    data_frames.append(df)

# 合并所有数据
df = pd.concat(data_frames, ignore_index=True)

# 显示数据基本信息
print("\n数据基本信息:")
print(f"数据集形状: {df.shape}")
print("\n数据类型:")
print(df.dtypes)

print("\n数据前5行:")
print(df.head())

print("\n基本统计信息:")
print(df.describe())

print("\n缺失值信息:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({
    '缺失值数量': missing_values,
    '缺失百分比': missing_percentage.round(2)
})
print(missing_info[missing_info['缺失值数量'] > 0])

# 数据可视化

# 创建输出目录
os.makedirs('visualizations', exist_ok=True)

print("\n开始生成可视化图表...")

print("生成数值特征分布直方图...")
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("生成相关性热图...")
if len(numeric_columns) > 1:
    plt.figure(figsize=(12, 10))
    corr_matrix = df[numeric_columns].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('feature correlation heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()

print("生成箱线图...")
if len(numeric_columns) > 0:
    selected_columns = numeric_columns[:min(6, len(numeric_columns))]
    
    fig = make_subplots(rows=len(selected_columns), cols=1, 
                        subplot_titles=[f'{col} boxplot' for col in selected_columns])
    
    for i, column in enumerate(selected_columns):
        fig.add_trace(
            go.Box(y=df[column].dropna(), name=column),
            row=i+1, col=1
        )
    
    fig.update_layout(height=400*len(selected_columns), width=900, showlegend=False)
    fig.write_html('visualizations/boxplots.html')


categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
if len(categorical_columns) > 0:
    print("generate category variable count plot...")
    selected_cat_columns = categorical_columns[:min(3, len(categorical_columns))]
    
    for column in selected_cat_columns:
        value_counts = df[column].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'{column} category distribution (top 10)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'visualizations/{column}_counts.png')
        plt.close()

print("\n数据探索性分析完成。可视化结果保存在 'visualizations' 目录中。")