import requests
import time
from concurrent.futures import ThreadPoolExecutor
import statistics
import argparse
from utils.json_util import load_list_from_json,read_parquet_to_list
import json
from tqdm import tqdm
from utils.json_util import save_data_to_json

#NUM_WORKERS = 120
#TOTAL_REQUESTS = 500
import re
from typing import List, Dict
def parse_tools_from_retrieval_content(retrieval_content):
    """
    从 retrieval_content 中全局提取 category_name, tool_name, api_name。
    参数
    ----
    retrieval_content : str 或 dict（dict时有"result"键）
    返回
    ----
    List[Dict[str, str]]，每个元素形如：
        {
            "category_name": "...",
            "tool_name": "...",
            "api_name": "..."
        }
    """
    if isinstance(retrieval_content, dict):
        text = retrieval_content.get("result", "")
    else:
        text = retrieval_content
    text = text.replace("\\n", "\n")
    pattern = re.compile(
        r"category\s*_?\s*name\s*:\s*([^\n\r]+).*?tool\s*_?\s*name\s*:\s*([^\n\r]+).*?api\s*_?\s*name\s*:\s*([^\n\r]+)",
        re.IGNORECASE | re.DOTALL
    )
    results = []
    for match in pattern.finditer(text):
        category_name = match.group(1).strip()
        tool_name = match.group(2).strip()
        api_name = match.group(3).strip()
        if category_name and tool_name and api_name:
            results.append({
                "category_name": category_name,
                "tool_name": tool_name,
                "api_name": api_name
            })
    return results

def send_request(request_item):
    URL = f"http://127.0.0.1:{request_item['port']}/retrieve"
    payload = {
        "category": None,
        "query": request_item["question"],
        "topk": request_item["topk"],
        "return_scores": True
    }
    
    start = time.time()
    try:
        response = requests.post(URL, json=payload, timeout=300)
        latency = (time.time() - start) * 1000
        #print(response.json())
        response_list = response.json()['result'][0]
     
        api_list = []
        for api_response in response_list:
            api = parse_tools_from_retrieval_content(api_response['api_doc'])[0]
            api_name = api['category_name'] + "." + api['tool_name'] + "." + api['api_name']
            api_list.append(api_name)

        return {
            'success': response.status_code == 200,
            'latency': latency,
            'status': response.status_code,
            'response': api_list
        }
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {
            'success': False,
            'latency': latency,
            'error': str(e)
        }


def retrieve(args):
    #1. load test data
    data_list = read_parquet_to_list(args.test_data_file_path)
    print(len(data_list))
    data_result_list = []
    for data in tqdm(data_list):
        request_item = {
            'index': data['extra_info']['index'],
            'question': data['extra_info']['question'],
            'ground_truth': data['reward_model']['ground_truth'],
            'topk': args.topk,
            'port': args.port
        }
        #input(data)
        result = send_request(request_item)
        if not result['success']:
            print(result)
            input("press")
        response_item = {
            'index': data['extra_info']['index'],
            'data_source': data['data_source'],
            'question': data['extra_info']['question'],
            'ground_truth': data['reward_model']['ground_truth'],
            'selected_apis': result['response']
        }
        data_result_list.append(response_item)

    save_file_path = f"{args.save_file_path}/{args.save_file_name}.json"
    save_data_to_json(data_result_list, save_file_path)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="retrieve based on different retriever.")
    parser.add_argument("--port",default=1370,help="port of url.",)
    parser.add_argument("--topk",default=10,help="topk of retrieval",)
    parser.add_argument("--test_data_file_path",default="./data/stabletoolbench_dataset/tool_selection.parquet")
    parser.add_argument("--save_file_path",default="./experiment_results/baseline")
    parser.add_argument("--save_file_name",default="toolbench_IR_bert")
    args = parser.parse_args()

    retrieve(args)

    # results = []

    # with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]
    #     for future in futures:
    #         results.append(future.result())
