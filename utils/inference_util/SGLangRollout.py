from verl.workers.rollout.sglang_rollout import SGLangRollout
from verl.utils.fs import copy_to_local
import hydra
from verl.utils import hf_processor,hf_tokenizer
from transformers import AutoConfig
import ray
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
import torch.distributed as dist
import torch
import os
from  test.test_verl_utils.utils_sglang import initialize_global_process_group, clean_torchelastic_env
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
import pandas as pd
import numpy as np
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.dataset.rl_dataset import collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from utils.verl_util.show_message import print_messages
from utils.inference_util.structure_util import process_batch_for_structure
from utils.json_util import save_data_to_json

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    # this is for local ray cluster
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(), #{"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))
    
@ray.remote(num_cpus=1)
def main_task(config):
    print(">>> start to infer")
    infer(config)

def get_val_dataloader(val_dataset, val_batch_size, collate_fn, num_workers = 16):
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return val_dataloader
    
def construct_input_data(test_batch, tokenizer, config, debug = False):
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
    if "multi_modal_data" in test_batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("multi_modal_data")
    if "raw_prompt" in test_batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("raw_prompt")
    if "tools_kwargs" in test_batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("tools_kwargs")
    if "interaction_kwargs" in test_batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("interaction_kwargs")
    if "agent_name" in test_batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("agent_name")
    test_gen_batch = test_batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )

    test_gen_batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
        "validate": True,
    }

    if debug:
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}") 
    return test_gen_batch
        
def inference(val_dataloader, tokenizer, config, actor_rollout_wg, val_reward_fn = None, debug = True):
    # Lists to collect samples for the table
    sample_inputs = []
    inference_results = []
    count = 0
    for test_data in tqdm(val_dataloader, desc='Val Batch'):
        #print(f">>> start {count} to generate")
        test_batch = DataProto.from_single_dict(test_data)
        count += 1

        # Store original inputs
        input_ids = test_batch.batch["input_ids"]
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        sample_inputs.extend(input_texts)

        test_gen_batch = construct_input_data(test_batch, tokenizer, config)

        # pad to be divisible by dp_size
        size_divisor = actor_rollout_wg.world_size
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
        #generate
        test_output_gen_batch_padded = actor_rollout_wg.generate_sequences(test_gen_batch_padded)
        # unpad
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

        # Store generated outputs
        test_batch = test_batch.union(test_output_gen_batch)
        test_batch.meta_info["validate"] = True
        results = process_batch_for_structure(test_batch.non_tensor_batch)
        inference_results.extend(results)
        if debug:
            print("-----------test_output_gen_batch-----------")
            print(test_batch.non_tensor_batch)
            print("-----------test_output_gen_batch-----------")
            print(results)     
            #print_messages(test_batch.non_tensor_batch)
            #input()
            break

        torch.cuda.empty_cache()
    #trainer.project_name='toolsearcher' \
    #trainer.experiment_name="$EXPERIMENT_NAME" \
    save_data_path = f"./experiment_results/{config.trainer.project_name}/{config.trainer.experiment_name}.json"
    save_data_to_json(inference_results, save_data_path)

def infer(
    config,
    trust_remote_code=False,
):  

    print(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
    
    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    val_dataset = RLHFDataset(
        data_files=config.data.val_files,
        tokenizer=tokenizer,
        processor=processor,
        config=config.data,
    )
    print("=========val_dataset[0]=========")
    #字典格式化输出
    for key, value in val_dataset[0].items():
        print(f"{key}:{value}")
    print("data_len", len(val_dataset))
    
    
    val_dataloader = get_val_dataloader(val_dataset, val_batch_size = config.data.val_batch_size, collate_fn = collate_fn)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config.actor_rollout_ref, role="actor_rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    actor_rollout_wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        profile_option=config.trainer.npu_profile.options,
    )
    actor_rollout_wg.init_model()
    inference(val_dataloader, tokenizer, config, actor_rollout_wg, debug= False)    

    
if __name__ == "__main__":
    main()