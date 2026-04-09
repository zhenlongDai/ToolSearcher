"""
PlanThenSelection Inference Pipeline

This baseline method follows a multi-step approach:
1. Plan: Generate a step-by-step plan for solving the task
2. Parse steps: Extract individual steps from the plan
3. Retrieve per step: Retrieve candidate APIs for each step
4. Selection: Select appropriate APIs based on plan and step-wise candidates
"""

import argparse
from dataclasses import dataclass, fields
import sys
import multiprocessing
from multiprocessing import Process, Manager, Lock, Barrier
import os
import re
import copy
import gc
import requests
from typing import Any, List, Dict, Optional
from tqdm import tqdm

from utils.vllm_util.configs import InferenceConfig, load_config
from utils.vllm_util.inferBase import InferenceBase, init_os_env, get_part_eval_dataset
from utils.json_util import read_parquet_to_list, save_list_to_json
from utils.file_util import read_file
from utils.string_util import render_template
from utils.evaluation_util.format_util import parse_tool_apiname_lists_from_retrieval_content
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PlanThenSelection Inference Configuration")
    parser.add_argument('--config', type=str, help="YAML config file path",
                        default='baseline_method/PlanThenSelection/configs/plan_then_selection.yaml')
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


def retrieval_topk_apis_content(query: str, topk: int, port: int) -> str:
    """Retrieve top-k candidate APIs via HTTP request"""
    url = f"http://127.0.0.1:{port}/retrieve"
    payload = {
        "category": None,
        "query": query,
        "topk": topk,
        "return_scores": True
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response_list = response.json()['result'][0]
        apis_content = ""
        for index, response_item in enumerate(response_list):
            apis_content += f"doc {index + 1}: {response_item['api_doc']}\n"
        return apis_content
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""


def parse_plan_steps(plan_text: str) -> List[str]:
    """Parse plan text to extract individual steps"""
    steps = []

    # Try to match "Step N: ..." pattern
    step_pattern = r'Step\s*\d+\s*[:：]\s*(.+?)(?=Step\s*\d+\s*[:：]|$)'
    matches = re.findall(step_pattern, plan_text, re.DOTALL | re.IGNORECASE)

    if matches:
        for match in matches:
            step_text = match.strip()
            if step_text:
                steps.append(step_text)

    # Fallback: split by newlines if no step pattern found
    if not steps:
        lines = plan_text.split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines and plan tags
            if line and not line.startswith('<') and not line.startswith('#'):
                # Remove common prefixes like "1.", "-", "*", etc.
                cleaned = re.sub(r'^[\d\.\-\*\•\)]+\s*', '', line)
                if cleaned and len(cleaned) > 10:  # Minimum meaningful step length
                    steps.append(cleaned)

    # If still no steps, use the whole plan as one step
    if not steps:
        # Remove plan tags
        cleaned_plan = re.sub(r'</?plan>', '', plan_text).strip()
        if cleaned_plan:
            steps.append(cleaned_plan)

    return steps


def format_step_retrieved_content(steps: List[str], retrieved_contents: List[str]) -> str:
    """Format retrieved content for each step"""
    formatted = ""
    for i, (step, content) in enumerate(zip(steps, retrieved_contents), 1):
        formatted += f"## Step {i}: {step}\n\n"
        formatted += f"Candidate APIs:\n{content}\n\n"
    return formatted


SYSTEM_CONTENT = "You are a super intelligent AI assistant that achieves my day-to-day tasks completely autonomously by interacting with apps/tools using their associated APIs on my behalf."


class InferencePipeline(InferenceBase):
    """Multi-step inference pipeline: Plan -> Parse Steps -> Retrieve per Step -> Selection"""

    def __init__(self, args):
        # Get extra config fields from args.extra
        extra = getattr(args, 'extra', {})
        self.retrieval_port = extra.get('retrieval_port', 1350)
        self.retrieval_topk = extra.get('retrieval_topk', 100)
        self.plan_max_output_tokens = extra.get('plan_max_output_tokens', 1500)
        self.step_topk = extra.get('step_topk', 20)  # Top-k for each step
        self.plan_prompt_template = read_file(extra.get('plan_prompt_path', './baseline_method/PlanThenSelection/prompt_template/plan_prompt.txt'))
        self.selection_prompt_template = read_file(extra.get('selection_prompt_path', './baseline_method/PlanThenSelection/prompt_template/selection_prompt.txt'))
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # Call parent init (this will call load_eval_dataset)
        super().__init__(args)

        print("data_list[0]:", self.data_list[0])
        print("------prompt----------\n", self.data_list[0]['prompt'], "\n---------------------")

    def get_prompt(self, messages, tokenizer):
        """Apply chat template to messages"""
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text

    def truncate_messages(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """Truncate messages to fit within max_tokens"""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(tokens) <= max_tokens:
            return messages

        # Truncate the last user message content
        new_messages = list(messages)
        last_msg = dict(new_messages[-1])
        kept_ids = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(kept_ids, skip_special_tokens=True)
        # Estimate content portion
        last_msg["content"] = truncated_text[-min(len(last_msg["content"]), len(truncated_text) * 2):]
        new_messages[-1] = last_msg
        return new_messages

    def load_eval_dataset(self, args):
        """Load dataset and prepare prompts for stage 1 (Plan)"""
        data_file_path = args.data_file_path
        data_list = read_parquet_to_list(data_file_path)

        if self.debug_mode:
            data_list = data_list[:20]

        prompt_data_list = []
        for data in tqdm(data_list, desc="Processing data"):
            question = data['extra_info']['question']

            # Build plan prompt (Stage 1) - no retrieval at this stage
            user_content = render_template(
                self.plan_prompt_template,
                question=question
            )
            messages = [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": user_content}
            ]
            messages = self.truncate_messages(messages, args.max_input_tokens)
            prompt = self.get_prompt(messages, self.tokenizer)

            new_data = {
                'index': data['extra_info']['index'],
                'data_source': data['data_source'],
                'prompt': prompt,
                'question': question,
                'ground_truth': data['reward_model']['ground_truth'],
            }
            prompt_data_list.append(new_data)

        return prompt_data_list

    def process_outputs(self, outputs, part_prompt_list, part_eval_infos):
        """Process outputs - not used in this pipeline, logic in generation()"""
        pass

    def generation(self, data_list, gen_args, dist_args, result_queue, lock, barrier, debug_mode=True):
        """Override generation to implement multi-step retrieval inference"""
        model_path, temperature, top_p, gpu_memory_utilization, dtype, max_input_tokens, max_output_tokens, use_lora, lora_path = self.get_specific_args(gen_args)
        dp_size, tp_size, local_dp_rank, dp_master_ip, dp_master_port = self.get_distributed_args(dist_args)

        init_os_env(local_dp_rank, dp_size, dp_master_ip, dp_master_port)
        part_eval_infos, part_prompt_list = get_part_eval_dataset(data_list, local_dp_rank, dp_size, lock)

        print(f"DP rank {local_dp_rank} start loading model......")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enforce_eager=True,
            enable_lora=use_lora
        )

        # ========== Stage 1: Generate Plans ==========
        print(f"DP rank {local_dp_rank} Stage 1: Generating plans......")
        sampling_params_plan = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.plan_max_output_tokens
        )
        outputs_plan = llm.generate(part_prompt_list, sampling_params_plan)

        # Process plan outputs: parse steps and retrieve for each step
        stage2_prompts = []
        plan_results = []
        step_retrieved_results = []

        for i, output in enumerate(outputs_plan):
            plan_text = output.outputs[0].text
            question = part_eval_infos[i]['question']

            # Parse steps from plan
            steps = parse_plan_steps(plan_text)
            #print(f"  Sample {i}: Parsed {len(steps)} steps from plan")

            # Retrieve candidate APIs for each step
            step_retrieved_contents = []
            step_search_apis = []  # Store parsed API names for each step
            for step_idx, step in enumerate(steps):
                # Use step description as query for retrieval
                retrieved = retrieval_topk_apis_content(step, self.step_topk, self.retrieval_port)
                step_retrieved_contents.append(retrieved)
                # Parse API names from retrieved content
                api_names = parse_tool_apiname_lists_from_retrieval_content(retrieved)
                step_search_apis.append(api_names)

            # Format retrieved content for all steps
            step_content_formatted = format_step_retrieved_content(steps, step_retrieved_contents)

            # Build selection prompt (Stage 2)
            user_content = render_template(
                self.selection_prompt_template,
                question=question,
                plan=plan_text,
                step_retrieved_content=step_content_formatted
            )
            messages = [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": user_content}
            ]
            messages = self.truncate_messages(messages, max_input_tokens)
            prompt = self.get_prompt(messages, self.tokenizer)

            stage2_prompts.append({"prompt": prompt})
            plan_results.append(plan_text)
            step_retrieved_results.append({
                'steps': steps,
                'retrieved_contents': step_retrieved_contents,
                'search_apis': step_search_apis  # Add parsed API names
            })

        # ========== Stage 2: Select APIs based on Plans and Step-wise Candidates ==========
        print(f"DP rank {local_dp_rank} Stage 2: Selecting APIs based on plans and step-wise candidates......")
        sampling_params_selection = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens
        )
        outputs_selection = llm.generate(stage2_prompts, sampling_params_selection)

        # Combine results
        result = []
        for i, output in enumerate(outputs_selection):
            generated_text = output.outputs[0].text
            new_data = {
                'index': copy.deepcopy(part_eval_infos[i]['index']),
                'data_source': copy.deepcopy(part_eval_infos[i]['data_source']),
                'question': copy.deepcopy(part_eval_infos[i]['question']),
                'plan': copy.deepcopy(plan_results[i]),
                'parsed_steps': copy.deepcopy(step_retrieved_results[i]['steps']),
                'search_apis': copy.deepcopy(step_retrieved_results[i]['search_apis']),
                'generated_text': copy.deepcopy(generated_text),
            }
            result.append(new_data)

        print(">>> Generation completed")
        result_queue.put(result)
        print(">>> Put in result queue")
        barrier.wait()
        print(f"Exit the process {local_dp_rank}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    print(args)

    inference_pipeline = InferencePipeline(args)
    inference_pipeline.run_in_parpallel()
    print("finished\n")