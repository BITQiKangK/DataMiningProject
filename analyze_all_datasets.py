import os
import pandas as pd
import matplotlib.pyplot as plt
from analyze_dataset import analyze_dataset
import time

# 数据集路径
datasets = [
    "dataset/1G_data",
    "dataset/10G_data",
    "dataset/30G_data"
]

# 输出目录
output_dirs = [
    "high_value_analysis_1G_data",
    "high_value_analysis_10G_data",
    "high_value_analysis_30G_data"
]

# 存储分析结果
results = []

# 分析所有数据集
for i, (dataset_path, output_dir) in enumerate(zip(datasets, output_dirs)):
    print(f"\n{'='*80}")
    print(f"开始分析数据集 {i+1}/3: {dataset_path}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        dataset_size, high_value_percentage = analyze_dataset(dataset_path, output_dir)
        end_time = time.time()
        duration = end_time - start_time
        
        results.append({
            'dataset': os.path.basename(dataset_path),
            'size': dataset_size,
            'high_value_percentage': high_value_percentage,
            'time_taken': duration
        })
        
        print(f"数据集 {dataset_path} 分析完成，耗时 {duration:.2f} 秒")
    except Exception as e:
        print(f"分析数据集 {dataset_path} 时出错: {e}")

# 生成比较报告
if results:
    # 创建对比结果目录
    comparison_dir = "datasets_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 生成结果表格
    results_df = pd.DataFrame(results)
    results_df['time_taken'] = results_df['time_taken'].apply(lambda x: f"{x:.2f} 秒")
    results_df.columns = ['数据集', '记录数量', '高价值用户百分比', '处理时间']
    
    results_df.to_csv(f"{comparison_dir}/comparison_results.csv", index=False)
    
    # 生成比较图表
    plt.figure(figsize=(12, 6))
    
    # 绘制数据集大小与高价值用户百分比的关系
    df_plot = pd.DataFrame(results)
    plt.scatter(df_plot['size'], df_plot['high_value_percentage'], s=100)
    
    for i, row in df_plot.iterrows():
        plt.annotate(row['dataset'], 
                     (row['size'], row['high_value_percentage']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    
    plt.xlabel('数据集大小（记录数）')
    plt.ylabel('高价值用户百分比')
    plt.title('数据集大小与高价值用户百分比关系')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/dataset_size_vs_high_value_percentage.png")
    
    # 生成中文比较报告
    with open(f"{comparison_dir}/数据集对比分析.md", 'w', encoding='utf-8') as f:
        f.write("# 不同规模数据集的高价值用户分析对比\n\n")
        
        f.write("## 分析概述\n\n")
        f.write("本报告对比了三个不同大小的数据集（1G、10G和30G）中高价值用户识别的结果。")
        f.write("通过K-均值聚类算法分析，我们识别了每个数据集中的高价值用户群体，并比较了数据集规模对结果的影响。\n\n")
        
        f.write("## 数据集基本信息\n\n")
        f.write("| 数据集 | 记录数量 | 高价值用户百分比 | 处理时间 |\n")
        f.write("|--------|----------|-----------------|----------|\n")
        
        for result in results:
            f.write(f"| {result['dataset']} | {result['size']} | {result['high_value_percentage']:.2f}% | {result['time_taken']:.2f}秒 |\n")
        
        f.write("\n## 主要发现\n\n")
        
        # 找出高价值用户百分比最高的数据集
        highest_percentage = max(results, key=lambda x: x['high_value_percentage'])
        lowest_percentage = min(results, key=lambda x: x['high_value_percentage'])
        
        f.write(f"1. **高价值用户比例**：在所有数据集中，{highest_percentage['dataset']} 数据集显示最高的高价值用户比例，为 {highest_percentage['high_value_percentage']:.2f}%\n")
        
        # 比较数据集规模对结果的影响
        if len(results) > 1:
            if abs(results[0]['high_value_percentage'] - results[-1]['high_value_percentage']) < 5:
                f.write("2. **数据集规模影响**：不同规模的数据集显示出相似的高价值用户比例，这表明我们的方法具有良好的稳定性\n")
            else:
                f.write("2. **数据集规模影响**：不同规模的数据集显示出不同的高价值用户比例，这表明更大的数据集可能提供更准确的用户细分\n")
        
        # 计算处理时间与数据集大小的关系
        if len(results) > 1:
            size_ratio = results[-1]['size'] / results[0]['size']
            time_ratio = results[-1]['time_taken'] / results[0]['time_taken']
            
            if time_ratio > size_ratio:
                f.write(f"3. **计算效率**：随着数据集大小增加，处理时间增长更快（规模增加{size_ratio:.1f}倍，时间增加{time_ratio:.1f}倍）\n")
            else:
                f.write(f"3. **计算效率**：随着数据集大小增加，处理时间增长相对较慢（规模增加{size_ratio:.1f}倍，时间仅增加{time_ratio:.1f}倍）\n")
        
        f.write("\n## 建议\n\n")
        f.write("1. **统一分析框架**：建议采用相同的分析方法对不同规模的数据集进行分析，以确保结果的可比性\n")
        f.write("2. **增量分析**：对于大型数据集，可以考虑采用增量分析方法，定期更新高价值用户群体\n")
        f.write("3. **维度选择**：在聚类分析中，应重点关注影响用户价值的关键特征，如收入、消费频率等\n")
        
        f.write("\n## 结论\n\n")
        f.write("通过对比不同规模数据集的分析结果，我们可以得出以下结论：\n\n")
        
        if len(results) > 1:
            if abs(results[0]['high_value_percentage'] - results[-1]['high_value_percentage']) < 5:
                f.write("* 高价值用户的占比在不同数据集间相对稳定，这表明我们的识别方法具有较高的可靠性\n")
            else:
                f.write("* 不同数据集显示出不同的高价值用户占比，这表明数据集的规模和分布对结果有显著影响\n")
        
        f.write("* 无论数据集大小如何，通过聚类分析都能有效识别出具有共同特征的高价值用户群体\n")
        f.write("* 对于业务决策，建议综合考虑不同规模数据集的分析结果，以获得更全面的用户价值洞察\n")
    
    # 生成英文比较报告
    with open(f"{comparison_dir}/datasets_comparison_report.md", 'w', encoding='utf-8') as f:
        f.write("# High-Value User Analysis Comparison Across Different Dataset Sizes\n\n")
        
        f.write("## Analysis Overview\n\n")
        f.write("This report compares the results of high-value user identification across three datasets of different sizes (1G, 10G, and 30G). ")
        f.write("Using K-means clustering analysis, we identified high-value user groups in each dataset and compared how dataset scale affects the results.\n\n")
        
        f.write("## Dataset Information\n\n")
        f.write("| Dataset | Number of Records | High-Value User Percentage | Processing Time |\n")
        f.write("|---------|-------------------|----------------------------|----------------|\n")
        
        for result in results:
            f.write(f"| {result['dataset']} | {result['size']} | {result['high_value_percentage']:.2f}% | {result['time_taken']:.2f}s |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # 找出高价值用户百分比最高的数据集
        highest_percentage = max(results, key=lambda x: x['high_value_percentage'])
        lowest_percentage = min(results, key=lambda x: x['high_value_percentage'])
        
        f.write(f"1. **High-Value User Proportion**: Among all datasets, {highest_percentage['dataset']} showed the highest percentage of high-value users at {highest_percentage['high_value_percentage']:.2f}%\n")
        
        # 比较数据集规模对结果的影响
        if len(results) > 1:
            if abs(results[0]['high_value_percentage'] - results[-1]['high_value_percentage']) < 5:
                f.write("2. **Impact of Dataset Scale**: Different sized datasets showed similar proportions of high-value users, indicating good stability in our methodology\n")
            else:
                f.write("2. **Impact of Dataset Scale**: Different sized datasets showed varying proportions of high-value users, suggesting larger datasets may provide more accurate user segmentation\n")
        
        # 计算处理时间与数据集大小的关系
        if len(results) > 1:
            size_ratio = results[-1]['size'] / results[0]['size']
            time_ratio = results[-1]['time_taken'] / results[0]['time_taken']
            
            if time_ratio > size_ratio:
                f.write(f"3. **Computational Efficiency**: As dataset size increased, processing time grew faster (size increased {size_ratio:.1f}x, time increased {time_ratio:.1f}x)\n")
            else:
                f.write(f"3. **Computational Efficiency**: As dataset size increased, processing time grew relatively slower (size increased {size_ratio:.1f}x, time increased only {time_ratio:.1f}x)\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("1. **Unified Analysis Framework**: We recommend using the same analysis methods across different dataset sizes to ensure comparability of results\n")
        f.write("2. **Incremental Analysis**: For large datasets, consider using incremental analysis methods to periodically update high-value user groups\n")
        f.write("3. **Feature Selection**: In clustering analysis, focus on key features that impact user value, such as income and purchase frequency\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("By comparing analysis results across different dataset sizes, we can draw the following conclusions:\n\n")
        
        if len(results) > 1:
            if abs(results[0]['high_value_percentage'] - results[-1]['high_value_percentage']) < 5:
                f.write("* The proportion of high-value users remains relatively stable across different datasets, indicating the reliability of our identification method\n")
            else:
                f.write("* Different datasets show varying proportions of high-value users, indicating that dataset scale and distribution significantly impact results\n")
        
        f.write("* Regardless of dataset size, clustering analysis effectively identifies high-value user groups with common characteristics\n")
        f.write("* For business decisions, we recommend considering analysis results from different dataset sizes to gain more comprehensive insights into user value\n")
    
    print(f"\n比较分析完成。结果已保存至 {comparison_dir} 目录")
else:
    print("没有足够的结果生成比较报告") 