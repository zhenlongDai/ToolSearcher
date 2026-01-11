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

BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]

def print_d(d_list):
  d_str_list = ""
  d_str_n_list = ""
  for d in d_list:
    str_d = str(d)
    d_n = dump_yaml(d)
    str_d_n = str(d_n)
    print("---", len(str_d))
    d_str_list += str_d
    print("+++", len(str_d_n))
    d_str_n_list += str_d_n
    print(">>>")
  return d_str_list, d_str_n_list

if __name__ == "__main__":
  local_path = '/ossfs/workspace/hy65/dzl/model/Qwen2.5-7B-Instruct'
  processor = hf_tokenizer(local_path, trust_remote_code=False)
  d1 = {'category_name': 'Data', 'tool_name': 'fake_users', 'api_name': 'user', 'api_description': '[tool description]:fake users is a Api that give you fake users [api description]:get one user', 'template_response': {'results': [{'gender': 'str', 'name': {'title': 'str', 'first': 'str', 'last': 'str'}, 'location': {'street': {'number': 'int', 'name': 'str'}, 'city': 'str', 'state': 'str', 'country': 'str', 'postcode': 'int', 'coordinates': {'latitude': 'str', 'longitude': 'str'}, 'timezone': {'offset': 'str', 'description': 'str'}}, 'email': 'str', 'login': {'uuid': 'str', 'username': 'str', 'password': 'str', 'salt': 'str', 'md5': 'str', 'sha1': 'str', 'sha256': 'str'}, 'dob': {'date': 'str', 'age': 'int'}, 'registered': {'date': 'str', 'age': 'int'}, 'phone': 'str', 'cell': 'str', 'id': {'name': 'str', 'value': 'NoneType'}, 'picture': {'large': 'str', 'medium': 'str', 'thumbnail': 'str'}, 'nat': 'str', '_list_length': 1}], 'info': {'seed': 'str', 'results': 'int', 'page': 'int', 'version': 'str'}}}
  d2 = {'category_name': 'Data', 'tool_name': 'fake_users', 'api_name': 'get_user_by_gender', 'api_description': '[tool description]:fake users is a Api that give you fake users [api description]:get user by gender', 'required_parameters': [{'name': 'gender', 'type': 'STRING', 'description': '', 'default': 'male'}], 'template_response': {'results': [{'gender': 'str', 'name': {'title': 'str', 'first': 'str', 'last': 'str'}, 'location': {'street': {'number': 'int', 'name': 'str'}, 'city': 'str', 'state': 'str', 'country': 'str', 'postcode': 'int', 'coordinates': {'latitude': 'str', 'longitude': 'str'}, 'timezone': {'offset': 'str', 'description': 'str'}}, 'email': 'str', 'login': {'uuid': 'str', 'username': 'str', 'password': 'str', 'salt': 'str', 'md5': 'str', 'sha1': 'str', 'sha256': 'str'}, 'dob': {'date': 'str', 'age': 'int'}, 'registered': {'date': 'str', 'age': 'int'}, 'phone': 'str', 'cell': 'str', 'id': {'name': 'str', 'value': 'str'}, 'picture': {'large': 'str', 'medium': 'str', 'thumbnail': 'str'}, 'nat': 'str', '_list_length': 1}], 'info': {'seed': 'str', 'results': 'int', 'page': 'int', 'version': 'str'}}}
  d3 = {'category_name': 'Data', 'tool_name': 'random_user_by_api_ninjas', 'api_name': 'v1_randomuser', 'api_description': '[tool description]:Random user data generator for placeholders and testing. See more info at https://api-ninjas.com/api/randomuser [api description]:API Ninjas Random User API endpoint. Returns a fake random user profile.', 'template_response': {'username': 'str', 'sex': 'str', 'address': 'str', 'name': 'str', 'email': 'str', 'birthday': 'str'}}
  d4 = {'category_name': 'Data', 'tool_name': 'feku_json', 'api_name': 'getuserbyid', 'api_description': '[tool description]:Free Feku ( Fake ) API for Testing and Prototyping. [api description]:To to Specific User by ID', 'required_parameters': [{'name': 'id', 'type': 'string', 'description': '', 'default': '1'}], 'template_response': {'id': 'int', 'firstName': 'str', 'lastName': 'str', 'email': 'str', 'phone': 'str', 'website': 'str'}}
  d5 = {'category_name': 'Data', 'tool_name': 'uers_api', 'api_name': 'get_all_users', 'api_description': '[tool description]:Fake users data for Employee Management [api description]:Get all the users', 'template_response': {'data': [{'Email': 'str', 'Image': 'str', 'LastLogin': 'str', 'Name': 'str', 'Role': 'str', 'Status': 'bool', 'id': 'str', '_list_length': 12}]}}
  d_list = [d1]
  d_str_list, d_str_n_list = print_d(d_list)
 
  raw_prompt= d_str_list
  print("A.......")
  model_inputs = processor(text=[raw_prompt], return_tensors="pt")
  model_inputs = dict(model_inputs)
  print(model_inputs['input_ids'].shape)
  input_ids = model_inputs['input_ids'][0]          # shape: [1118]
  inputs_text = input_ids[:512]      
  print(type(inputs_text))
  print(len(inputs_text))
  new_text = processor.decode(inputs_text, skip_special_tokens = True)
  print(new_text)


  print("B......")
  model_inputs = processor(text=[d_str_n_list], return_tensors="pt")
  model_inputs = dict(model_inputs)
  print(model_inputs['input_ids'].shape)
  input_ids = model_inputs['input_ids'][0]          # shape: [1118]
  inputs_text = input_ids[:512]      
  print(type(inputs_text))
  print(len(inputs_text))
  new_text = processor.decode(inputs_text, skip_special_tokens = True)
  print(new_text)