import argparse
import os
import tempfile
from utils.file_util import read_file
from utils.string_util import render_template
import pandas as pd
from typing import Any, Literal, cast
from utils.json_util import save_list_to_parquet, load_list_from_json
from utils.file_util import get_all_json_file_name
from utils.toolbench_util.format_util import get_toolbench_prompt_categories
from utils.toolbench_util.format_util import standardize_category, standardize
from utils.file_util import ensure_directory
import json
system_content = "You are a super intelligent AI assistant that achieves my day-to-day tasks completely autonomously by interacting with apps/tools using their associated APIs on my behalf."


def get_name(data):
    #cat toolname and api_name
    name = standardize(data['tool_name']) + "_" + standardize(data['api_name'])
    return name

def get_relevant_set(relevant_APIs):
    #元素为「app_name」+「api_name」
    relevant_set = set()
    for api in relevant_APIs:
        api_name = standardize(api[1])
        toolapi_name = standardize(api[0]) + "_" + api_name
        if toolapi_name not in relevant_set:
            relevant_set.add(toolapi_name)
    return relevant_set

def fliter_relevant_api_list(api_list, relevant_apis):
    #确保api_list中有relevant APIs中的元素，保留relevant APIs中的元素，删除其他元素
    new_api_list = []
    relevant_apis_set = get_relevant_set(relevant_apis)
    for api_item in api_list:
        toolapi_name = get_name(api_item)
        if toolapi_name in relevant_apis_set:
            new_api_list.append(api_item)
    return new_api_list

def constcut_api_ground_truth(onedata):
    api_list = fliter_relevant_api_list(onedata['api_list'], onedata['relevant APIs']) #important since test data is different with training data in format for inference.
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
        "api_doc_search_tool": {
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
    # print(oneData)
    # input(extra_info)

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
    


def process_stabletoolbench_parquet_data(args):
    prompt_template_path = args.prompt_template_path
    prompt_template = cast(str, read_file(prompt_template_path.replace("/", os.sep)))
    data_source_tags = get_all_json_file_name(args.origin_data_dir)
    print(data_source_tags)
    prompt_categories = get_toolbench_prompt_categories()
  
    data_list = []
    for data_source_tag in data_source_tags:
        
        data_source_tag_name = data_source_tag.split(".json")[0]
        json_file_path = os.path.join(args.origin_data_dir, f"{data_source_tag_name}.json")
        json_list = load_list_from_json(json_file_path)
        print(f"json_list: {len(json_list)}")
        for oneData in json_list:
            new_data = process_single_data(oneData, prompt_template, data_source_tag_name, prompt_categories)
            if new_data is not None:
                data_list.append(new_data)
        

    print(f"data_list: {len(data_list)}")
    print("----------data[0]-----------")
    print_qarquet_item(data_list[0])
    print("----------data[0]-----------")
    
    # 运行调试
    #debug_dataframe_types(train_data_list)
    save_list_to_parquet(data_list, os.path.join(args.save_local_dir, f"{args.save_file_name}.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process dataset and save to Parquet.")
    parser.add_argument("--origin_data_dir",default="./experiment/stabletoolbench_experiment/StableToolBench/solvable_queries/test_instruction",help="Local directory to load the original Json files.",)
    parser.add_argument("--save_local_dir",default="./data/stabletoolbench_dataset",help="Local directory to save the processed Parquet files.",)
    parser.add_argument("--prompt_template_path", default="./construct/process_data/prompt_template/api_search_prompt.txt", help="prompt_template_path")
    parser.add_argument("--save_file_name",default="tool_selection",help="Local directory to save the processed Parquet files.",)

    args = parser.parse_args()
    print(args)
    process_stabletoolbench_parquet_data(args)
