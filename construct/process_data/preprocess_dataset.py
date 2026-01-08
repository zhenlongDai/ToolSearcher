import argparse
import os
import tempfile
from utils.file_util import read_file
from utils.string_util import render_template
import pandas as pd
from typing import Any, Literal, cast
from utils.json_util import save_list_to_parquet, load_list_from_json
from utils.toolbench_util.format_util import get_toolbench_prompt_categories
from utils.toolbench_util.format_util import standardize_category, standardize
from utils.file_util import ensure_directory
import json
system_content = "You are a super intelligent AI assistant that achieves my day-to-day tasks completely autonomously by interacting with apps/tools using their associated APIs on my behalf."





def constcut_api_ground_truth(onedata):
    api_list = onedata['api_list']
    api_ground_truth = []
    for api in api_list:
        category_name = standardize_category(api['category_name'])
        tool_name = standardize(api['tool_name'])
        api_name = standardize(api['api_name'])
        api_ground_truth.append(f"{category_name}.{tool_name}.{api_name}")
    return api_ground_truth

def process_single_data(oneData, prompt_template, data_source_tag, prompt_categories):
    question = oneData['query']
    if not isinstance(question, str):
        return None 
    user_content= render_template(
                prompt_template,
                tool_categories=prompt_categories,
                question=question,
            )
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
    
    ground_truth = constcut_api_ground_truth(oneData)

    reward_model = {
                    "style": "rule",
                    "ground_truth": ground_truth
                }
    # Build tools kwargs structure
    tools_kwargs = {
        "search_tool_docs": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tag}
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": oneData['query_id'],
        "need_tools_kwargs": True,
        "question": question,
        "split": None,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tag,
            "prompt": prompt,
            "reward_model": reward_model,
            "extra_info": extra_info,
            "metadata": None,
        }
    )

def print_qarquet_item(data):
    data = data.to_dict()
    print(json.dumps(data, indent=4, ensure_ascii=False))
    
def process_split_data(data_list):
    split_train = []
    split_eval = []
    split_count_map = {}
    for data in data_list:
        api_count = len(data['reward_model']['ground_truth'])
        if api_count > 5:
            split_eval.append(data)
        elif api_count <= 3:
            if api_count not in split_count_map:
                split_count_map[api_count] = 1
            else:
                split_count_map[api_count] += 1
            
            if split_count_map[api_count] <= 50:
                split_eval.append(data)
            else:
                split_train.append(data)
        else:
            split_train.append(data)
            
    return split_train, split_eval

def process_toolbench_parquet_data(args):
    prompt_template_path = args.prompt_template_path
    prompt_template = cast(str, read_file(prompt_template_path.replace("/", os.sep)))
    data_source_tags = ['G1', 'G2', 'G3']
    prompt_categories = get_toolbench_prompt_categories()
    train_data_list = []
    eval_data_list = []
    for data_source_tag in data_source_tags:
        json_file_path = os.path.join(args.origin_data_dir, f"{data_source_tag}_query.json")
        json_list = load_list_from_json(json_file_path)
        data_list = []
        for oneData in json_list:
            new_data = process_single_data(oneData, prompt_template, data_source_tag, prompt_categories)
            if new_data is not None:
                data_list.append(new_data)
        split_train, split_eval = process_split_data(data_list)
        print(f"split_train: {len(split_train)}, split_eval: {len(split_eval)}")
        train_data_list.extend(split_train)
        eval_data_list.extend(split_eval)
    print(f"train_data_list: {len(train_data_list)}")
    print(f"eval_data_list: {len(eval_data_list)}")
    print("----------data-----------")
    print_qarquet_item(train_data_list[0])
    print("----------data-----------")
    
    # 运行调试
    #debug_dataframe_types(train_data_list)
    save_list_to_parquet(train_data_list, os.path.join(args.save_local_dir, "train.parquet"))
    save_list_to_parquet(eval_data_list, os.path.join(args.save_local_dir, "eval.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process dataset and save to Parquet.")
    parser.add_argument("--origin_data_dir",default="./data/plan_data_v1",help="Local directory to load the original Json files.",)
    parser.add_argument("--save_local_dir",default="./data/toolplan_qarquet_data",help="Local directory to save the processed Parquet files.",)
    parser.add_argument("--prompt_template_path", default="./construct/process_data/prompt_template/prompt.txt", help="prompt_template_path")

    args = parser.parse_args()

    process_toolbench_parquet_data(args)
