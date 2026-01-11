from transformers import AutoConfig
from verl.utils import hf_processor,hf_tokenizer
import torch.distributed as dist
import torch
import pandas as pd
import numpy as np
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from utils.verl_util.show_message import print_messages
from utils.json_util import dump_yaml
from typing import List, Union
from transformers import AutoTokenizer

def print_d(d_list):
  d_str_list = ""
  d_str_n_list = ""
  i = 0
  for d in d_list:
    str_d = str(d)
    d_n = dump_yaml(d)
    str_d_n = str(d_n)
    print("---", len(str_d))
    i += 1
    d_str_list += f"doc {i}:\n" + str_d
    print("+++", len(str_d_n))
    d_str_n_list += f"doc {i}:\n" + str_d_n
    print(">>>")
  return d_str_list, d_str_n_list

def truncate_api_docs(
    standard_API_docs: Union[str, List[str]],
    retrieval_model_path: str,
    max_len: int = 768
) -> Union[str, List[str]]:
    """
    按 tokenizer 的 token 长度截断 API 文档内容。
    参数：
        standard_API_docs: 字符串或字符串列表
        retrieval_model_path: 用于加载 tokenizer 的模型路径/名称
        max_len: 最大 token 长度（含 special tokens）
    返回：
        截断后的字符串或字符串列表（与输入类型相同）
    """
    tokenizer = AutoTokenizer.from_pretrained(retrieval_model_path)
    def _truncate_one(text: str) -> str:
        truncated = False
        # 编码为 token id（不返回 tensor，避免维度问题）
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]  # List[int]
        if len(input_ids) == max_len:
          truncated = True 
        # 直接 decode 截断后的 token
        truncated_text = tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        if truncated:
          truncated_text += "...[truncate]"
        return truncated_text
    if isinstance(standard_API_docs, str):
        return _truncate_one(standard_API_docs)
    else:
        return [_truncate_one(doc) for doc in standard_API_docs]

if __name__ == "__main__":
  local_path = "/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B"
  processor = hf_tokenizer(local_path, trust_remote_code=False)
  d1 = {'category_name': 'Data', 'tool_name': 'fake_users', 'api_name': 'user', 'api_description': '[tool description]:fake users is a Api that give you fake users [api description]:get one user', 'template_response': {'results': [{'gender': 'str', 'name': {'title': 'str', 'first': 'str', 'last': 'str'}, 'location': {'street': {'number': 'int', 'name': 'str'}, 'city': 'str', 'state': 'str', 'country': 'str', 'postcode': 'int', 'coordinates': {'latitude': 'str', 'longitude': 'str'}, 'timezone': {'offset': 'str', 'description': 'str'}}, 'email': 'str', 'login': {'uuid': 'str', 'username': 'str', 'password': 'str', 'salt': 'str', 'md5': 'str', 'sha1': 'str', 'sha256': 'str'}, 'dob': {'date': 'str', 'age': 'int'}, 'registered': {'date': 'str', 'age': 'int'}, 'phone': 'str', 'cell': 'str', 'id': {'name': 'str', 'value': 'NoneType'}, 'picture': {'large': 'str', 'medium': 'str', 'thumbnail': 'str'}, 'nat': 'str', '_list_length': 1}], 'info': {'seed': 'str', 'results': 'int', 'page': 'int', 'version': 'str'}}}
  d2 = {'category_name': 'Data', 'tool_name': 'fake_users', 'api_name': 'get_user_by_gender', 'api_description': '[tool description]:fake users is a Api that give you fake users [api description]:get user by gender', 'required_parameters': [{'name': 'gender', 'type': 'STRING', 'description': '', 'default': 'male'}], 'template_response': {'results': [{'gender': 'str', 'name': {'title': 'str', 'first': 'str', 'last': 'str'}, 'location': {'street': {'number': 'int', 'name': 'str'}, 'city': 'str', 'state': 'str', 'country': 'str', 'postcode': 'int', 'coordinates': {'latitude': 'str', 'longitude': 'str'}, 'timezone': {'offset': 'str', 'description': 'str'}}, 'email': 'str', 'login': {'uuid': 'str', 'username': 'str', 'password': 'str', 'salt': 'str', 'md5': 'str', 'sha1': 'str', 'sha256': 'str'}, 'dob': {'date': 'str', 'age': 'int'}, 'registered': {'date': 'str', 'age': 'int'}, 'phone': 'str', 'cell': 'str', 'id': {'name': 'str', 'value': 'str'}, 'picture': {'large': 'str', 'medium': 'str', 'thumbnail': 'str'}, 'nat': 'str', '_list_length': 1}], 'info': {'seed': 'str', 'results': 'int', 'page': 'int', 'version': 'str'}}}
  d3 = {'category_name': 'Data', 'tool_name': 'random_user_by_api_ninjas', 'api_name': 'v1_randomuser', 'api_description': '[tool description]:Random user data generator for placeholders and testing. See more info at https://api-ninjas.com/api/randomuser [api description]:API Ninjas Random User API endpoint. Returns a fake random user profile.', 'template_response': {'username': 'str', 'sex': 'str', 'address': 'str', 'name': 'str', 'email': 'str', 'birthday': 'str'}}
  d4 = {'category_name': 'Data', 'tool_name': 'feku_json', 'api_name': 'getuserbyid', 'api_description': '[tool description]:Free Feku ( Fake ) API for Testing and Prototyping. [api description]:To to Specific User by ID', 'required_parameters': [{'name': 'id', 'type': 'string', 'description': '', 'default': '1'}], 'template_response': {'id': 'int', 'firstName': 'str', 'lastName': 'str', 'email': 'str', 'phone': 'str', 'website': 'str'}}
  d5 = {'category_name': 'Data', 'tool_name': 'uers_api', 'api_name': 'get_all_users', 'api_description': '[tool description]:Fake users data for Employee Management [api description]:Get all the users', 'template_response': {'data': [{'Email': 'str', 'Image': 'str', 'LastLogin': 'str', 'Name': 'str', 'Role': 'str', 'Status': 'bool', 'id': 'str', '_list_length': 12}]}}
  d_list = [d1,d2,d3,d4,d5]
  d_str_list, d_str_n_list = print_d(d_list)
 
  print(d_str_n_list)
  print("------")
  standard_API_doc = truncate_api_docs(d_str_n_list, local_path)
  print(standard_API_doc)
 