import pandas as pd


def statistic_stabletoolbench(file_path):
    df = pd.read_parquet(file_path)

    print("=" * 60)
    print("数据集统计报告")
    print("=" * 60)

    # 统计字典
    stats = {}
    all_lengths = []  # 收集所有API数量用于总体统计

    for idx, row in df.iterrows():
        kv = row.to_dict()
        try:
            apis_ground_truth = kv['reward_model']['ground_truth']
            data_source = kv['data_source']
            gt_len = len(apis_ground_truth)
        except Exception:
            continue

        if data_source not in stats:
            stats[data_source] = []
        stats[data_source].append(gt_len)
        all_lengths.append(gt_len)

    # 总体统计
    print(f"\n【总体统计】")
    print(f"  数据总量: {len(df)}")
    if all_lengths:
        print(f"  API数量 - 最大值: {max(all_lengths)}, 最小值: {min(all_lengths)}, 均值: {sum(all_lengths)/len(all_lengths):.2f}")

    # 按数据源分组统计
    print("\n【按数据源分组统计】")
    for ds in sorted(stats.keys()):
        lengths = stats[ds]
        if not lengths:
            continue
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = sum(lengths) / len(lengths)
        print(f"\n  {ds}:")
        print(f"    数据量: {len(lengths)}")
        print(f"    API数量 - 最大值: {max_len}, 最小值: {min_len}, 均值: {avg_len:.2f}")

    print("\n" + "=" * 60)


# 用法示例:
if __name__ == "__main__":
    statistic_stabletoolbench('./data/stabletoolbench_dataset/tool_selection.parquet')