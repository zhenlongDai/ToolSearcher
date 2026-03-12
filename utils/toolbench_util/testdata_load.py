import argparse
import os
import pandas as pd
from typing import Any, Literal, cast
from utils.json_util import load_list_from_json
from utils.file_util import get_all_json_file_name
import json

def process_stabletoolbench_data(args):

  data_source_tags = get_all_json_file_name(args.origin_data_dir)
  print(data_source_tags)
  data_list = []
  for data_source_tag in data_source_tags:
      data_source_tag_name = data_source_tag.split(".json")[0]
      json_file_path = os.path.join(args.origin_data_dir, f"{data_source_tag_name}.json")
      json_list = load_list_from_json(json_file_path)
      print(f"json_list: {len(json_list)}")
      data_list.extend(json_list)
  print(data_list[0])   
  return data_list
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process dataset and save to Parquet.")
    parser.add_argument("--origin_data_dir",default="./experiment/stabletoolbench_experiment/StableToolBench/solvable_queries/test_instruction",help="Local directory to load the original Json files.",)
    # parser.add_argument("--save_local_dir",default="./data/stabletoolbench_dataset",help="Local directory to save the processed Parquet files.",)
    # parser.add_argument("--prompt_template_path", default="./construct/process_data/prompt_template/prompt.txt", help="prompt_template_path")
    # parser.add_argument("--save_file_name",default="tool_selection",help="Local directory to save the processed Parquet files.",)

    args = parser.parse_args()
    print(args)
    process_stabletoolbench_data(args)