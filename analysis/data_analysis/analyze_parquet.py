"""
分析 train.parquet 数据集
- 统计 G1/G2/G3 数据数量
- 分析相关 API 数量的最大值、最小值、均值
"""

import pandas as pd
import numpy as np


def analyze_parquet(file_path: str):
    df = pd.read_parquet(file_path)

    print("=" * 60)
    print("数据集分析报告")
    print("=" * 60)

    # 1. 统计 G1/G2/G3 数据数量
    print("\n【G1/G2/G3 数据数量统计】")
    data_source_counts = df['data_source'].value_counts().sort_index()
    for source, count in data_source_counts.items():
        print(f"  {source}: {count}")
    print(f"  总计: {len(df)}")

    # 2. 分析相关 API 数量
    print("\n【相关 API 数量分析】")

    # 提取每条数据的 API 数量
    api_counts = []
    for idx, row in df.iterrows():
        try:
            extra_info = row['extra_info']
            if extra_info and isinstance(extra_info, dict):
                tools_kwargs = extra_info.get('tools_kwargs', {})
                api_doc_search_tool = tools_kwargs.get('api_doc_search_tool', {})
                create_kwargs = api_doc_search_tool.get('create_kwargs', {})
                ground_truth = create_kwargs.get('ground_truth', None)
                if ground_truth is not None:
                    if isinstance(ground_truth, np.ndarray):
                        api_counts.append(len(ground_truth))
                    elif isinstance(ground_truth, (list, tuple)):
                        api_counts.append(len(ground_truth))
                    else:
                        api_counts.append(0)
                else:
                    api_counts.append(0)
            else:
                api_counts.append(0)
        except Exception as e:
            api_counts.append(0)

    api_counts = np.array(api_counts)

    print(f"  最大值: {api_counts.max()}")
    print(f"  最小值: {api_counts.min()}")
    print(f"  均值: {api_counts.mean():.2f}")
    print(f"  中位数: {np.median(api_counts):.2f}")

    # 3. 按 G1/G2/G3 分组统计 API 数量
    print("\n【按数据源分组的 API 数量统计】")
    df['api_count'] = api_counts
    grouped = df.groupby('data_source')['api_count']

    for source in sorted(df['data_source'].unique()):
        group_data = grouped.get_group(source)
        print(f"\n  {source}:")
        print(f"    数据量: {len(group_data)}")
        print(f"    API数量 - 最大值: {group_data.max()}, 最小值: {group_data.min()}, 均值: {group_data.mean():.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    file_path = "/ossfs/workspace/hy65/dzl/code/toolPlaner/data/toolplan_qarquet_data/train.parquet"
    analyze_parquet(file_path)