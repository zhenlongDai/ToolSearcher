import argparse
import os
import tempfile
from utils.file_util import read_file
from utils.string_util import render_template
import pandas as pd
from typing import Any, Literal, cast
from utils.json_util import save_list_to_parquet, load_list_from_json, load_data_from_json
from utils.appworld_util.format_util import categories
from utils.toolbench_util.format_util import standardize_category, standardize
from utils.file_util import ensure_directory
import json
system_content = "You are an AI Assistant. Your task is to analyze a given complex user request and determine which available APIs would be useful to accomplish it autonomously on behalf of the user (supervisor)."


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
    required_apis = onedata['required_apis']
    if isinstance(required_apis, list):
        api_ground_truth = []
        for api in required_apis:    
            api_ground_truth.append(api)
    else:
        api_ground_truth = []
 
        
    return api_ground_truth

def process_single_data(oneData, prompt_template, data_source_tag, api_descriptions_string):
    question = oneData['instruction']
    if not isinstance(question, str):
        return None 
    user_content= render_template(
                prompt_template,
                required_apis_string=api_descriptions_string,
                instruction=question,
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
        "index": oneData['index'],
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
    


def process_appworld_parquet_data(args):
    prompt_template_path = args.prompt_template_path
    prompt_template = cast(str, read_file(prompt_template_path.replace("/", os.sep)))

    content_dict = load_data_from_json(args.content_path)
    api_descriptions_string = content_dict['api_descriptions_string']
 
    data_list = []
    json_list = load_list_from_json(args.origin_data_path)
    print(f"json_list: {len(json_list)}")
    for oneData in json_list:
        data_source_tag_name = oneData['data_scoure']
        new_data = process_single_data(oneData, prompt_template, data_source_tag_name, api_descriptions_string)
        
        if new_data is not None:
            data_list.append(new_data)
        

    print(f"data_list: {len(data_list)}")
    print("----------data[0]-----------")
    print_qarquet_item(data_list[0])
    print("----------data[0]-----------")
    
    save_list_to_parquet(data_list, args.save_local_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process dataset and save to Parquet.")
    parser.add_argument("--origin_data_path",default="./data/appworld_dataset/tool_selection.json",help="Local directory to load the original Json files.",)
    parser.add_argument("--save_local_path",default="./data/appworld_dataset/apipredictor.parquet",help="Local directory to save the processed Parquet files.",)
    parser.add_argument("--prompt_template_path", default="./construct/process_data/prompt_template/appworld/api_predictor.txt", help="prompt_template_path")
    parser.add_argument("--content_path", default="./data/appworld_dataset/api_predictor_content.json", help="api_predictor_content")

    args = parser.parse_args()
    print(args)
    process_appworld_parquet_data(args)
