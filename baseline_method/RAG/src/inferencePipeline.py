import argparse
from utils.vllm_util.configs import InferenceConfig, load_config
from typing import Any, List, Dict
import sys
import multiprocessing
import os
from time import sleep
import copy
import gc
from utils.vllm_util.inferBase import InferenceBase, init_os_env, get_part_eval_dataset
from dataclasses import fields
from utils.json_util import read_parquet_to_list
from transformers import AutoTokenizer
from tqdm import tqdm

def parse_args():
    """解析命令行参数并返回配置实例，命令行参数会覆盖配置文件中的值"""
    parser = argparse.ArgumentParser(description="训练配置")
    parser.add_argument('--config', type=str, help="YAML 配置文件路径", default='config.yaml')
    args, _ = parser.parse_known_args()
    config = load_config(args.config)  
    for field in fields(config):
        parser.add_argument(
            f'--{field.name}',
            type=type(getattr(config, field.name)),
            default=getattr(config, field.name),
        )

    args = parser.parse_args()
    return args

class InferencePipeline(InferenceBase):
    
  def __init__(self, args):
    super().__init__(args)

    # other args
    #self.system_prompt = args.system_prompt
   
    
    #self.save_file_path = args.save_file_path
    print("data_list[0]:", self.data_list[0])
    print("------prompt----------\n" , self.data_list[0]['prompt'], "\n---------------------")

  def truncate_messages(self, messages: List[Dict], tokenizer, max_tokens: int) -> List[Dict]:
    """Truncate messages to fit within max_tokens"""
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
      return messages

    # Truncate the last user message content
    new_messages = list(messages)
    last_msg = dict(new_messages[-1])
    kept_ids = tokens[:max_tokens]
    truncated_text = tokenizer.decode(kept_ids, skip_special_tokens=True)
    # Estimate content portion
    last_msg["content"] = truncated_text[-min(len(last_msg["content"]), len(truncated_text) * 2):]
    new_messages[-1] = last_msg
    return new_messages

  def get_prompt(self, messages, tokenizer):
    # Prepare the input to the model
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return text
    
  def load_eval_dataset(self, args):
    data_file_path = args.data_file_path
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data_list = read_parquet_to_list(data_file_path)

    if self.debug_mode:
      data_list = data_list[:20]

    prompt_data_list  = []
    for data in tqdm(data_list, desc="Processing data"):
        new_data = {
          'index': data['extra_info']['index'],
          'data_source': data['data_source'],
        }
        # Truncate messages to fit within max_input_tokens
        messages = self.truncate_messages(data['messages'], tokenizer, args.max_input_tokens)
        new_data['prompt'] = self.get_prompt(messages, tokenizer)
        prompt_data_list.append(new_data)
    return prompt_data_list

  def process_outputs(self, outputs, part_prompt_list, part_eval_infos):
    result = []
    for i, output in enumerate(outputs):
      # prompt = output.prompt
      # if prompt != part_eval_infos[i]['prompt']:
      #     raise ValueError(f"Prompt: {prompt!r} not equal to output prompt: {part_eval_infos[i]['prompt']!r}") 
      generated_text = output.outputs[0].text
      new_data = {
                  'index': copy.deepcopy(part_eval_infos[i]['index']), 
                  'data_source': copy.deepcopy(part_eval_infos[i]['data_source']),
                  'generated_text': copy.deepcopy(generated_text)
                }
      result.append(new_data)

    return result
    
    
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parse_args()  # 解析命令行参数并加载配置
    print(args)
    #input()
    inference_pipeline = InferencePipeline(args)
    inference_pipeline.run_in_parpallel()  # 启动推理管道
    print("finished\n")
    #inference_pipeline.test_in_parallel()  # 启动推理管道