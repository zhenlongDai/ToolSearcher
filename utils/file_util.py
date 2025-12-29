### 1. 根据目录获取目录下一级别的所有文件夹名字path
import os
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
    