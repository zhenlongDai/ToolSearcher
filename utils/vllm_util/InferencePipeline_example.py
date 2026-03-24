import argparse
from utils.vllm_util.configs import InferenceConfig, load_config
from typing import Any
import sys
import multiprocessing
import os
from time import sleep
import copy
import gc
from utils.vllm_util.inferBase import InferenceBase, init_os_env, get_part_eval_dataset
from dataclasses import fields

def parse_args():
    """解析命令行参数并返回配置实例，命令行参数会覆盖配置文件中的值"""
    parser = argparse.ArgumentParser(description="训练配置")
    parser.add_argument('--config', type=str, help="YAML 配置文件路径", default='config.yaml')
    parser.add_argument('--prompt_mode', type=str, help="prompt_mode", default='test_prompt')
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
    self.system_prompt = args.system_prompt
    self.prompt_mode = args.prompt_mode
    #self.save_file_path = args.save_file_path
    print("data_list[0]:", self.data_list[0])
    print("------prompt----------\n" , self.data_list[0]['prompt'], "\n---------------------")
      #input()

  def load_eval_dataset(self, args):
    base_prompts = [
        "Summarize the following article:",
        "Translate the sentence into English:",
        "Explain the main idea of this paragraph:",
        "Generate a title for the given text:",
        "Rewrite the sentence to be more formal:",
        "List three key points from the passage:",
        "Answer the question based on the context:",
        "Classify the sentiment of the following review:",
        "Correct the grammar in this sentence:",
        "Provide a short description of the topic:",
    ]
    # 给每条 prompt 一个唯一 id，这里用整数 1..N
    data_list = []
    for i, prompt in enumerate(base_prompts, start=1):
        data_list.append({
            "index": i,
            "prompt": prompt,
        })

    return data_list 


  def process_outputs(self, outputs, part_prompt_list, part_eval_infos):
    result = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        if prompt != part_eval_infos[i]['prompt']:
            raise ValueError(f"Prompt: {prompt!r} not equal to output prompt: {part_eval_infos[i]['prompt']!r}") 
        
        generated_text = output.outputs[0].text
        new_data = { "generated_text": copy.deepcopy(generated_text), "index": copy.deepcopy(part_eval_infos[i]['index'])}
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