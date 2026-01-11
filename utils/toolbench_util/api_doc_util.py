
from utils.file_util import get_all_folder_name, get_all_json_file_name
import os
from utils.json_util import load_data_from_json
import re
import json
from utils.toolbench_util.format_util import standardize
from utils.json_util import dump_yaml

def extract_toolbench_api_docs(tool_dict, category_name: str = None):
    toolbench_api_docs = []
    tool_api_list = tool_dict.get('api_list', [])
    for tool_api in tool_api_list:
        tool_api['category_name'] = category_name if "category_name" not in tool_api else tool_api['category_name'] 
        tool_api['tool_name'] = tool_dict['tool_name']
        tool_api['tool_description'] =  "" if tool_dict.get('tool_description', "") == None else tool_dict.get('tool_description')
        toolbench_api_docs.append(tool_api)
    return toolbench_api_docs

def get_api_docs(category_name, category_tools_path: str) -> list[dict]:
    """
    get api docs of toolbench from category_name and category_tools_path
    :param category_name
    :param category_tools_path
    :return: all_api_docs_list, list of api docs
    """
    tool_json_path_list = get_all_json_file_name(category_tools_path, is_join_path=True)
    all_api_docs_list = []
    for tool_json_path in tool_json_path_list:
        tool_dic = load_data_from_json(tool_json_path)    
        tool_api_docs = extract_toolbench_api_docs(tool_dic, category_name = category_name)
        all_api_docs_list.extend(tool_api_docs)
    return all_api_docs_list

# 最终再处理结构信息

def clean_backslashes(text: str) -> str:
    """删除连续超过3个的反斜杠"""
    return re.sub(r'\\{4,}', '', text)

def clean_json_strings(obj):
    """
    递归处理JSON对象，清理所有字符串值中的连续反斜杠
    
    Args:
        obj: 任意JSON对象（dict, list, str, int, float, bool, None）
        
    Returns:
        处理后的对象
    """
    if isinstance(obj, dict):
        # 字典：递归处理所有key和value
        return {k: clean_json_strings(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        # 列表：递归处理所有元素
        return [clean_json_strings(item) for item in obj]
    
    elif isinstance(obj, str):
        # 字符串：清理反斜杠
        return clean_backslashes(obj)
    
    else:
        # 其他类型（int, float, bool, None）：直接返回
        return obj

def constrcut_toolbench_api_doc(api_doc):
    new_api_doc = {}

    new_api_doc['category_name'] = api_doc['category_name']
    new_api_doc['tool_name'] = standardize(api_doc['tool_name'])
    new_api_doc['api_name'] = standardize(api_doc['name'])
    new_api_doc['api_description'] = api_doc['description']
    if api_doc['tool_description'].strip() != "":
        new_api_doc['api_description'] = f"[tool]:{api_doc['tool_description']} " + "[api]:"+ api_doc['description']
    if api_doc['required_parameters'] != []:
        new_api_doc['required_parameters'] = api_doc['required_parameters']
    if api_doc['optional_parameters'] != []:
        new_api_doc['optional_parameters'] = api_doc['optional_parameters']
    if 'template_response' in api_doc:
        new_api_doc['template_response'] = api_doc['template_response']
    elif 'schema' in api_doc:
        new_api_doc['method'] = api_doc['schema']
    elif 'test_endpoint' in api_doc:
        new_api_doc['method'] = api_doc['test_endpoint']
    new_api_doc = clean_json_strings(new_api_doc)
    new_api_doc = dump_yaml(new_api_doc)
    return new_api_doc  

def constrcut_toolbench_api_docs(api_docs):
    new_api_docs = []
    for api_doc in api_docs:
        try:
            api_doc = constrcut_toolbench_api_doc(api_doc)
        except Exception as e:
            raise ValueError(f"error in constrcut_toolbench_api_doc, api_doc:{api_doc}, error:{e}")
            
        new_api_docs.append(api_doc)
    return new_api_docs