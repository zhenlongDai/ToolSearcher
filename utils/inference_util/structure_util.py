#This file is used for extracting and constructing structural object in evaluating the result of tool retriever 
from verl.utils.toolplan.search_process_util import parse_tools_from_retrieval_content
from utils.verl_util.show_message import process_batch
import re
import json

def parse_tool_list_v2(s):
    try:
        # 1. 提取花括号里面的内容
        m = re.search(r'\{.*\}', s, re.DOTALL)
        if not m:
            return []
        obj_str = m.group()
        # 2. 尝试用 json 解析，如果失败则做简单的字符串处理
        try:
            obj = json.loads(obj_str)
            # 兼容情况，可能 arguments 嵌套
            arguments = obj.get('arguments') if isinstance(obj.get('arguments'), dict) else obj
            tool_list_str = arguments.get('tool_list', '') if isinstance(arguments, dict) else ''
        except Exception:
            # 粗暴提取 tool_list 的内容
            m2 = re.search(r'"tool_list"\s*:\s*"([^"]*)"', obj_str)
            tool_list_str = m2.group(1) if m2 else ''
        # 3. 拆分字符串
        items = [i.strip() for i in tool_list_str.split(',') if i.strip()]
        return items
    except Exception:
        return []  # 不完整就返回空

def construct_api_dict_to_list(api_dict_list):
    api_list = []
    for value in api_dict_list:
        api_name = f"{value['category_name']}.{value['tool_name']}.{value['api_name']}"
        api_list.append(api_name)
    return api_list

def parse_tool_list(tool_list_str: str):
  if tool_list_str == None:
    return []
  
  return [
      item.strip()
      for item in tool_list_str.split(",")
      if item.strip()
  ]

def get_selected_apis_from_response(response_str):
    answer_pattern = r"<tool_list>(.*?)</tool_list>"
    match = re.finditer(answer_pattern, response_str, re.DOTALL)
    matches = list(match)
  
    tool_list_str = None
    # If there are 0  matches, return None
    if len(matches) < 1:
        tool_list_str = None 
    else:
        tool_list_str = matches[-1].group(1).strip()
    # If there are 2 or more matches, return the last one
    selected_api_list = parse_tool_list(tool_list_str)
    if len(selected_api_list) == 0:
        selected_api_list = parse_tool_list_v2(tool_list_str)
    return selected_api_list 

def parse_tools_from_turns(turns):
    search_apis = []
    selected_apis = []
    for turn in turns:
        if turn['role'] == 'user' or turn['role'] == 'system':
            continue
        elif turn['role'] == 'tool':
            dict_result_list= parse_tools_from_retrieval_content(turn['content'])
            api_list = construct_api_dict_to_list(dict_result_list)
            search_apis.append(api_list)
            
    last_turn = turns[-1]
    if last_turn['role'] == 'assistant':
        selected_apis = get_selected_apis_from_response(last_turn['content'])
        
    else:
        selected_apis = []

    return search_apis, selected_apis


def process_batch_for_structure(non_tensor_batch):  
  data_source_list = non_tensor_batch['data_source']
  extra_info_list =  non_tensor_batch['extra_info']
  structure_list, printer = process_batch(non_tensor_batch)
  results = []
  
  for structure, data_source, extra_info in zip(structure_list, data_source_list, extra_info_list):
      search_apis, selected_apis = parse_tools_from_turns(structure['turns'])
      result_item = {
          'data_source': data_source,
          'index': extra_info['index'],
          'question': extra_info['question'],
          'structure': structure,
          'search_apis': search_apis,
          'selected_apis': selected_apis,
          'ground_truth': extra_info['tools_kwargs']['api_doc_search_tool']['create_kwargs']['ground_truth'],
      }

      results.append(result_item)
  
  return results