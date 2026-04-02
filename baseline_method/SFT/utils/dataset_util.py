import re
import os
from utils.json_util import read_parquet_to_list
from transformers import AutoTokenizer
import json
from datasets import  Dataset, DatasetDict
from transformers import BertTokenizer
from tqdm import tqdm
# def load_json_as_hf_dataset(json_file_path, tokenizer, debug_mode=False):
#     data_list = read_parquet_to_list(json_file_path)
#     # 转换格式
#     processed_data = []
#     for data in data_list:
#         processed_data.append({
#             "data_source": data['data_source'],
#             "prompt":data['messages'][:2],
#             "solution": data['messages'][-1],
#         })
#     # 如果是调试模式，只取前10条数据
#     if debug_mode:
#         processed_data = processed_data[:100]
#     # 用 HuggingFace Dataset 构建
#     return Dataset.from_list(processed_data)


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


def load_json_as_hf_dataset(
    json_file_path,
    tokenizer,
    max_prompt_tokens: int = 4800,
    debug_mode: bool = False
):
    """
    读取 parquet -> HF Dataset

    - 不改变你原来返回的结构：
        {
            "data_source": ...,
            "prompt": messages[:2] (messages 格式，可能被截断),
            "solution": messages[-1] (原样保留)
        }
    - 只在内部对 prompt 做 token 截断，并统计被截断数量
    """
    data_list = read_parquet_to_list(json_file_path)

    processed_data = []
    truncated_count = 0
    if debug_mode:
        print(f"Debug mode: only processing first 100 samples.")
        data_list = data_list[:100]

    for data in tqdm(data_list, desc="Processing samples"):
        messages = data["messages"]
        #print(data)
        #input()
        # 截断前两条 message
        prompt_messages, truncated = truncate_prompt_messages(
            messages,
            tokenizer,
            max_prompt_tokens=max_prompt_tokens,
        )
        if truncated:
            truncated_count += 1

        # solution 保持原样：最后一条 message
        solution_message = messages[-1]

        processed_data.append({
            "data_source": data["data_source"],
            "prompt": prompt_messages,     # 仍然是 messages 列表
            "solution": solution_message,  # 单条 message dict
        })

    # 调试模式：只取前100条
    if debug_mode:
        processed_data = processed_data[:100]

    dataset = Dataset.from_list(processed_data)
    print(f"Truncated {truncated_count} samples due to prompt length exceeding {max_prompt_tokens} tokens.")
    return dataset


# def get_conditional_apis_from_prompt(prompt: str, solution_str: str):

def load_compeltion_dataset(dataset_file_path , tokenizer, max_prompt_tokens, debug_mode=False):
    """
    Load the dataset from the given file path.
    
    Args:
        dataset_file_path (str): Path to the dataset file.
        
    Returns:
        datasets.Dataset: Loaded dataset.
    """
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_file_path}")
    train_file_path = os.path.join(dataset_file_path, "train.parquet")
    dev_file_path = os.path.join(dataset_file_path, "dev.parquet")
    train_dataset = load_json_as_hf_dataset(train_file_path, tokenizer, max_prompt_tokens, debug_mode)
    dev_dataset = load_json_as_hf_dataset(dev_file_path, tokenizer, max_prompt_tokens, debug_mode)
    dataset = DatasetDict({
        "train": train_dataset,
        "test": dev_dataset
    })
        
    # Ensure the dataset has the required columns
    print(dataset["train"].column_names)
    if "prompt" not in dataset["train"].column_names or "solution" not in dataset["train"].column_names:
        raise ValueError("Dataset must contain 'prompt' and 'solution' columns.")
    
    return dataset

