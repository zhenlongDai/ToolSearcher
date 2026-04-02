from appworld.task import Task
from appworld import update_root,load_task_ids
import json
import torch
from tqdm import tqdm
import argparse
from utils.json_util import save_list_to_json
import os
from appworld.common.io import dump_yaml, read_file

def get_all_apis():
  task_ids = load_task_ids("train") 
  task_id = task_ids[0]
  task = Task.load(task_id, load_ground_truth=False)
  api_docs_json_list = []
  for app_name, api_docs in task.api_docs.items():
      for api_name, api_doc in api_docs.items():
          api_doc['category_name'] = app_name
          api_docs_json_list.append(api_doc)

  Lennum = len(api_docs_json_list)
  print(f"Total API docs: {Lennum}")
  return api_docs_json_list #[json.dumps(doc) for doc in api_docs_json_list]

def get_app_descriptions():
  task_ids = load_task_ids("train") 
  task_id = task_ids[0]
  task = Task.load(task_id, load_ground_truth=False)
  return task.app_descriptions

def constrcut_core_apis(required_apis):
  core_apis = []
  for api in required_apis:
    if "supervisor" in api or "login" in api:
      continue
    core_apis.append(api)
  return core_apis

def constrcut_appworld_dataset(task_names, save_file_path):
    data_list = []
    for task_name in task_names:
      task_ids = load_task_ids(task_name)  
      mode = "full" if task_name in ["train", "dev"] else "minimal"

      for task_id in tqdm(task_ids):
        task = Task.load(task_id, ground_truth_mode = mode)
        instruction = task.instruction
        supervisor = task.supervisor
        if mode == "full":
          required_apis = task.ground_truth.required_apis 
        else:
          required_apis = None
        item = {
          'index': task_id,
          'data_scoure': task_name,
          'instruction': instruction,
          'required_apis': required_apis,
          'core_apis': constrcut_core_apis(required_apis),
          'supervisor': {'first_name': supervisor['first_name'],
                         'last_name': supervisor['last_name'],
                         'email': supervisor['email'],
                         'phone_number': supervisor['phone_number']},
          'required_apps': task.ground_truth.required_apps,
          'metadata': { 'difficulty': task.ground_truth.metadata["difficulty"],
                        'num_apps': task.ground_truth.metadata["num_apps"],
                        'num_apis': task.ground_truth.metadata["num_apis"],
                      }
                        
        }
        data_list.append(item)
        
    save_list_to_json(data_list, save_file_path)


def construct_apipredictor_prompt():
      task_ids = load_task_ids("train") 
      task_id = task_ids[0]
      task = Task.load(task_id, load_ground_truth=False)

      api_descriptions = {
          app_name: {api_name: api_doc["description"] for api_name, api_doc in api_docs.items()}
          for app_name, api_docs in task.api_docs.items()
      }
      api_descriptions_string = dump_yaml(api_descriptions)
      app_descriptions = get_app_descriptions()
      save_object = {
        "app_descriptions": app_descriptions,
        "api_descriptions_string": api_descriptions_string
      }
      save_list_to_json(save_object, "data/appworld_dataset/api_predictor_content.json")

if __name__ == "__main__":

  update_root("./experiment/appworld_experiment/appworld")
  parser = argparse.ArgumentParser(description="process dataset and save to json.")
  parser.add_argument("--save_local_dir",default="./data/appworld_dataset",help="Local directory to save the processed json files.",)
  parser.add_argument("--mode",default="tool_selection", help="Mode to process the dataset. tool_selection/test")
  
  args = parser.parse_args()
   
  if args.mode == "tool_selection":
    task_names = ['train', 'dev'] 
  elif args.mode == "test":
    task_names = ['test_challenge', 'test_normal'] 
  
  save_file_path = os.path.join(args.save_local_dir, f"{args.mode}.json")
  
  #1.
  #constrcut_appworld_dataset(task_names, save_file_path)
  #get_app_descriptions()
  #2.
  #docs = get_all_apis()
  #save_file_path = os.path.join(args.save_local_dir, f"api_docs.json")
  #save_list_to_json(docs, save_file_path)

  #3.
  construct_apipredictor_prompt()