import pandas as pd

def check_parquet_data_count(parquet_path):
    """
    检查指定 parquet 文件的数据量（行数）。
    """
    try:
        df = pd.read_parquet(parquet_path)
        print(f'文件 {parquet_path} 的数据量为: {len(df)} 条')
        return len(df)
    except Exception as e:
        print(f'无法读取文件 {parquet_path}: {e}')
        return None

# 用法举例
parquet_file = '/ossfs/workspace/hy65/dzl/code/toolPlaner/data/stabletoolbench_dataset/tool_selection.parquet'
count = check_parquet_data_count(parquet_file)