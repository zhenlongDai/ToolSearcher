import pandas as pd

def statistic_stabletoolbench(file_path):
    df = pd.read_parquet(file_path)
    # 统计字典
    stats = {}

    for idx, row in df.iterrows():
        kv = row.to_dict()
        # 处理可能缺失的数据，容错
        try:
            apis_ground_truth = kv['reward_model']['ground_truth']
            data_source = kv['data_source']
            gt_len = len(apis_ground_truth)
        except Exception:
            continue

        if data_source not in stats:
            stats[data_source] = []
        stats[data_source].append(gt_len)

    # 统计每组的 max, min, avg
    for ds, lengths in stats.items():
        if not lengths:
            continue
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = sum(lengths) / len(lengths)
        print(f"data_source: {ds}")
        print(f"  Max: {max_len}")
        print(f"  Min: {min_len}")
        print(f"  Avg: {avg_len:.2f}")
        print('--------------------')

# 用法示例:
statistic_stabletoolbench('./data/stabletoolbench_dataset/tool_selection.parquet')