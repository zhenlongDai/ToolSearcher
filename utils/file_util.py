import os
from typing import Any, Literal, cast
import json
import hashlib
import pandas as pd
import numpy as np


class FileAlreadyExistsError(Exception):
    """自定义异常：文件已存在"""
    pass

def ensure_directory(file_path):
    """
    检查文件路径的目录是否存在，如果不存在则创建。
    
    :param file_path: 文件的完整路径
    """
    # 获取目录部分
    directory = os.path.dirname(file_path)
    
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def read_file(file_path: str, mode: Literal["r", "rb"] = "r") -> str | bytes:
    if "b" in mode:
        with open(file_path, mode=mode) as file:
            content_bytes = file.read()
        return cast(bytes, content_bytes)
    else:
        with open(file_path, mode=mode, encoding="utf-8") as file:
            content_str = file.read()
        return cast(str, content_str)
    
def get_all_folder_name(path, is_join_path=False):
    '''
    :param path: 目录
    :param is_join_path: 是否需要join path
    :return: 
    folder_name_list: 目录下下一级别的所有文件夹名字
    '''
    folder_name_list = []
    if not os.path.exists(path):
        print(f"错误：目录 {path} 不存在")
        return
    # 遍历第一级文件夹
    for folder_name in os.listdir(path):
        if is_join_path:
            folder_name = os.path.join(path, folder_name)
        folder_name_list.append(folder_name)
    return folder_name_list

def print_qarquet_item(data):
    data = data.to_dict()
    print(json.dumps(data, indent=4, ensure_ascii=False))
    
### 根据目录获取目录下所有json文件,
def get_all_json_file_name(folder_path, is_join_path=False):
    '''
    :param folder_path: 目录
    :param is_join_path: 是否需要join path
    :return: 
    json_file_name_list: 目录下一级别的所有json文件
    '''
    json_file_name_list = []
    if not os.path.exists(folder_path):
        print(f"错误：目录 {folder_path} 不存在")
        return
    # 遍历第一级文件夹
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            if is_join_path:
                file_name = os.path.join(folder_path, file_name)
            json_file_name_list.append(file_name)
    
    return json_file_name_list  


def get_all_file_names(directory_path):
    """
    获取指定路径下的所有文件名字

    :param directory_path: 目录的路径
    :return: 文件名字列表
    """
    try:
        # 列出目录下的所有文件和文件夹
        entries = os.listdir(directory_path)
        # 过滤掉文件夹，只保留文件
        file_names = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        return file_names
    
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for accessing the directory '{directory_path}'.")
        return []
    
def calculate_md5(input_string):
    """
    计算并返回字符串的MD5哈希值

    参数:
    input_string (str): 需要计算MD5哈希值的字符串

    返回:
    str: 输入字符串的MD5哈希值
    """
    # 创建一个md5哈希对象
    md5_hash = hashlib.md5()
    
    # 更新哈希对象并计算哈希值
    md5_hash.update(input_string.encode('utf-8'))
    
    # 返回十六进制哈希值
    return md5_hash.hexdigest()

def check_file_exists(filepath):
    """
    检查指定路径的文件是否存在。

    参数:
    filepath (str): 要检查的文件路径。

    返回:
    bool: 如果文件存在，返回True；否则，返回False。
    """
    return  os.path.isfile(filepath)


def save_pd_list_to_qarquet(data_list: list[pd.Series], file_path):
    df = pd.DataFrame(data_list)
    df.to_parquet(file_path)
    

def debug_dataframe_types(data_list: list[pd.Series]):
    df = pd.DataFrame(data_list)
    
    print("=" * 80)
    print("DataFrame 列信息:")
    print(df.dtypes)
    print("\n" + "=" * 80)
    
    # 检查 object 类型的列
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"\n列名: {col}")
            print("-" * 40)
            
            # 统计类型
            type_counts = df[col].apply(lambda x: type(x).__name__).value_counts()
            print("类型分布:")
            print(type_counts)
            
            # 如果是字典，检查键
            dict_samples = df[df[col].apply(lambda x: isinstance(x, dict))][col]
            if len(dict_samples) > 0:
                all_keys = set()
                for d in dict_samples:
                    all_keys.update(d.keys())
                print(f"\n字典的所有键: {all_keys}")
                
                # 检查每个键的值类型
                for key in all_keys:
                    value_types = dict_samples.apply(
                        lambda d: type(d.get(key)).__name__ if key in d else 'missing'
                    ).value_counts()
                    if len(value_types) > 1:
                        print(f"  ⚠️  键 '{key}' 的值类型不一致: {dict(value_types)}")