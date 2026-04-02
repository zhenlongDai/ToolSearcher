import argparse
import os
import tempfile
from utils.file_util import read_file
from utils.string_util import render_template
import pandas as pd
from typing import Any, Literal, cast
from utils.json_util import save_list_to_parquet, load_list_from_json, read_parquet_to_list
from utils.file_util import get_all_json_file_name
from utils.toolbench_util.format_util import get_toolbench_prompt_categories
from utils.toolbench_util.format_util import standardize_category, standardize
from utils.file_util import ensure_directory
from utils.evaluation_util.format_util import parse_tool_apiname_lists_from_retrieval_content
import json
from tqdm import tqdm 
import requests
from transformers import AutoTokenizer


system_content = "You are a super intelligent AI assistant that achieves my day-to-day tasks completely autonomously by interacting with apps/tools using their associated APIs on my behalf."


def send_request(request_item):
    URL = f"http://127.0.0.1:{request_item['port']}/retrieve"
    payload = {
        "category": None,
        "query": request_item["question"],
        "topk": request_item["topk"],
        "return_scores": True
    }
    
    response = requests.post(URL, json=payload, timeout=300)
    response_list = response.json()['result'][0]
    return response_list

def retrieval_topk_apis_content(query, topk, port):
    response_list = send_request({"question": query, "topk": topk, "port": port})
    apis_content = ""
    for index, response_item in enumerate(response_list):
        apis_content += f"doc {index+1}: " + response_item['api_doc'] + "\n"
    return apis_content

def print_qarquet_item(data):
    data = data.to_dict()
    print(data)
    #print(json.dumps(data, indent=4, ensure_ascii=False))
    

def process_single_data(infoData, prompt_template, data_source_tag, retrieved_content, max_prompt_tokens, tokenizer, have_answer=False):
    question = infoData['extra_info']['question']
    user_content = render_template(
                prompt_template,
                retrieved_content=retrieved_content,
                question=question,
            )
    
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    if tokenizer != None:
        messages, flag = truncate_prompt_messages(messages, tokenizer, max_prompt_tokens)

    if have_answer == True:
        #extract apis
        tool_apiname_lists = parse_tool_apiname_lists_from_retrieval_content(messages[-1]['content'])
      
        ground_truth_apis = infoData['reward_model']['ground_truth']
        answer_set = set(tool_apiname_lists) & set(ground_truth_apis)
        answer_list = list(answer_set)
        answer = construct_answer(answer_list)
        if len(answer_list) == 0:
            return None 
        # if len(answer_list) != len(ground_truth_apis):
        #     print(f"len:{len(tool_apiname_lists)}")
        #     print("tool_apiname_lists", tool_apiname_lists)
        #     print("answer_list", answer_list)
        #     print("ground_truth_apis", ground_truth_apis)
        #     print(answer)
        #     input()

       
        messages.append({"role": "assistant", "content": answer})

    reward_model = infoData['reward_model']
    extra_info = infoData['extra_info']
    extra_info['need_tools_kwargs'] = False
    extra_info['tools_kwargs'] = False
    
    return pd.Series(
        {
            "data_source": data_source_tag,
            "messages": messages,
            "reward_model": reward_model,
            "extra_info": extra_info,
            "metadata": None,
        }
    )


def construct_answer(groundtruth_apis):
    answer = "<tool_list>\n"
    Len = len(groundtruth_apis)
    for index, api in enumerate(groundtruth_apis):
        answer += api
        if index < Len - 1:
            answer += ","
    answer += "\n</tool_list>"
    return answer

def truncate_prompt_messages(messages, tokenizer, max_prompt_tokens: int):
    """
    只截断前两条 message (prompt 部分)，保留 messages 结构不变。
    返回: 截断后的 prompt_messages, 是否发生截断(bool)
    """

    prompt_messages = messages[:2]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    if len(prompt_ids) <= max_prompt_tokens:
        return prompt_messages, False

    kept_ids = prompt_ids[:max_prompt_tokens]
    truncated_text = tokenizer.decode(kept_ids, skip_special_tokens=True)

    new_prompt_messages = list(prompt_messages)
    last_msg = dict(new_prompt_messages[-1])
    last_msg["content"] = truncated_text
    new_prompt_messages[-1] = last_msg

    return new_prompt_messages, True



def process_parquet_data(args):
    prompt_template_path = args.prompt_template_path
    prompt_template = cast(str, read_file(prompt_template_path.replace("/", os.sep)))
    max_prompt_tokens = args.max_token_length
    if args.truncate_mode == "yes":
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path
        )
    else:
        tokenizer = None 
    info_data_list = read_parquet_to_list(args.origin_data_file)
  
    data_list = []
    for info_data in tqdm(info_data_list):
       
        data_source_tag_name = info_data['data_source']
        retrieved_content = retrieval_topk_apis_content(info_data['extra_info']['question'], args.topk, args.port)
        new_data = process_single_data(info_data, prompt_template, data_source_tag_name, retrieved_content, 
        max_prompt_tokens, tokenizer, args.data_mode != "input")

        if new_data is not None:
            data_list.append(new_data)
            
    print(f"data_list: {len(data_list)}")
    print("----------data[0]-----------")
    print_qarquet_item(data_list[0])
    print("----------data[0]-----------")
    
    save_list_to_parquet(data_list, os.path.join(args.save_local_dir, f"{args.save_file_name}.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process dataset and save to Parquet.")
    parser.add_argument("--origin_data_file",default="./data/stabletoolbench_dataset/tool_selection.parquet",help="Local directory to load the original Json files.",)
    #parser.add_argument("--method_mode",default="RAG",help="RAG")
    parser.add_argument("--data_mode",default="infullput",help="input/full")
    parser.add_argument("--prompt_template_path", default="./baseline_method/RAG/prompt_template/api_search_prompt.txt", help="prompt_template_path")
    parser.add_argument("--save_local_dir",default="./baseline_method/RAG/data/stabletoolbench_dataset",help="Local directory to save the processed Parquet files.",)
    parser.add_argument("--save_file_name",default="tool_selection",help="Local directory to save the processed Parquet files.")
    parser.add_argument("--port",default=1350,help="port of url.")
    parser.add_argument("--topk",default=35,help="topk of url.")
    parser.add_argument("--truncate_mode",default="no",help="no/yes")
    parser.add_argument("--max_token_length", default=4800, help="max token length.")
    parser.add_argument("--tokenizer_path", default="", help="tokenizer path")
    args = parser.parse_args()
    print(args)
    process_parquet_data(args)
