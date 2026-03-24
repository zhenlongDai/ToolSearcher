import argparse
from dataclasses import dataclass, fields
from utils.vllm_util.configs import InferenceConfig, load_config
from typing import Any
import sys
import multiprocessing
from multiprocessing import Process, Manager, Lock, Queue
from vllm.utils import get_open_port
import os
from time import sleep
from utils.json_util import save_list_to_json
from vllm import LLM, SamplingParams
import copy
import gc
from datasets import load_dataset
from vllm.lora.request import LoRARequest
from multiprocessing import Barrier
import torch


def init_os_env(local_dp_rank, dp_size, dp_master_ip, dp_master_port):    
    os.environ["VLLM_DP_RANK"] = str(local_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

def get_part_eval_dataset(data_list, local_dp_rank, dp_size, lock):
    # 计算每个进程处理的数据量
    data_per_rank = len(data_list) // dp_size
    start = local_dp_rank * data_per_rank
    end = start + data_per_rank
    if local_dp_rank == dp_size - 1:  # 最后一个进程处理剩余的数据
        end = len(data_list)
    # 获取当前进程需要处理的数据
    # 将data_list中start到end的数据分配给当前进程，新建list,避免进程卡死
    new_data_list = []
    with lock:
        for data in data_list[start:end]:
            new_data_list.append(copy.deepcopy(data))
    
    print(f"DP rank {local_dp_rank} needs to process {len(new_data_list)} prompts")

    part_prompt_list = [{"prompt": data['prompt'], "index": data['index']} for data in new_data_list]
    return new_data_list, part_prompt_list

class InferenceBase:
    
  def __init__(self, args):
    self.gen_args = {
            "model_path": args.model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "use_lora": args.use_lora,
            "lora_path": args.lora_path,
        }
    # distributed args
    self.dp_size = args.dp_size
    self.tp_size = args.tp_size
    self.dp_master_ip = "127.0.0.1"
    self.dp_master_port = get_open_port() if args.dp_master_port==0 else args.dp_master_port

    #other args
    self.debug_mode = args.debug_mode
    self.save_file_path = args.save_file_path

    data_list = self.load_eval_dataset(args)
    manager = Manager()
    self.data_list = manager.list(data_list)
    self.result_queue = manager.Queue()  # 用于存储每个进程的结果
    self.lock = Lock()  # 创建锁
    self.barrier = Barrier(self.dp_size)

  def load_eval_dataset(self, args):
    pass
  
  def process_outputs(self, outputs, part_prompt_list, part_eval_infos):
    pass

  def generation(self, data_list, gen_args, dist_args, result_queue, lock, barrier = None, debug_mode = True):
    model_path, temperature, top_p, gpu_memory_utilization, dtype, max_input_tokens, max_output_tokens, use_lora, lora_path = self.get_specific_args(gen_args)
    dp_size, tp_size, local_dp_rank, dp_master_ip, dp_master_port = self.get_distributed_args(dist_args)
    
    init_os_env(local_dp_rank, dp_size, dp_master_ip, dp_master_port)
    part_eval_infos, part_prompt_list = get_part_eval_dataset(data_list, local_dp_rank, dp_size, lock)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens = max_output_tokens)

    print(f"DP rank {local_dp_rank} start loading model......")
    llm = LLM(model=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype = dtype,
            enforce_eager=True,
            enable_lora=use_lora
            )
    
    print(f"rank {local_dp_rank} start generation......")
    if use_lora:
        print(f"DP rank {local_dp_rank} using LoRA with path: {lora_path}")
        outputs = llm.generate(part_prompt_list, 
                                sampling_params,
                                lora_request=LoRARequest(lora_name=f"lora_adapter_{local_dp_rank}",lora_int_id = local_dp_rank+1, lora_path=lora_path))
    else:
        outputs = llm.generate(part_prompt_list, sampling_params)
    
    result = self.process_outputs(outputs, part_prompt_list, part_eval_infos)
  
    print(">>> Generation completed")
    result_queue.put(result)
    print(">>> Put in result queue")
    barrier.wait()  # 等待所有进程到达此点
    print(f"Exit the process {local_dp_rank}")

  def get_specific_args(self, gen_args):
    model_path = gen_args["model_path"]
    temperature = gen_args["temperature"]
    top_p = gen_args["top_p"]
    gpu_memory_utilization = gen_args["gpu_memory_utilization"]
    dtype = gen_args["dtype"]
    max_input_tokens = gen_args["max_input_tokens"]
    max_output_tokens = gen_args["max_output_tokens"]
    use_lora = gen_args["use_lora"]
    lora_path = gen_args["lora_path"]
    return model_path, temperature, top_p, gpu_memory_utilization, dtype, max_input_tokens, max_output_tokens, use_lora, lora_path
  
  def get_distributed_args(self, dist_args):
    dp_size = dist_args["dp_size"]
    tp_size = dist_args["tp_size"]
    local_dp_rank = dist_args["local_dp_rank"]
    dp_master_ip = dist_args["dp_master_ip"]
    dp_master_port = dist_args["dp_master_port"]
    return dp_size, tp_size, local_dp_rank, dp_master_ip, dp_master_port
  
  def run_in_parpallel(self):

    procs = []
    for local_dp_rank in range(0, self.dp_size):
        print("local_dp_rank", local_dp_rank)
        dist_args = {
            "dp_size": self.dp_size,
            "tp_size": self.tp_size,
            "local_dp_rank": local_dp_rank,
            "dp_master_ip": self.dp_master_ip,
            "dp_master_port": self.dp_master_port,
        }
        
        proc = Process(target=self.generation,
                        args=(self.data_list,
                              self.gen_args,
                              dist_args,
                              self.result_queue, 
                              self.lock,  
                              self.barrier, 
                              self.debug_mode)
                        )
        proc.start()
        procs.append(proc)
    
    # 等待所有子进程结束
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            print(f"Process {proc.pid} exited with code {proc.exitcode}")

    # 从 result_queue 获取每个子进程的返回值
    results = []
    while not self.result_queue.empty():
        results.extend(self.result_queue.get())
        
    results = sorted(results, key=lambda x: x["index"])
    save_list_to_json(results, self.save_file_path)
    print("len(results)",len(results))
    print("results[0]:\n",results[0])

