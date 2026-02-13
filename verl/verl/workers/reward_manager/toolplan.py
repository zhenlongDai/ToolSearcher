# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.utils.toolplan.show_message import _structure_single_dialogue
from verl.utils.toolplan.search_process_util import search_process, parse_tools_from_retrieval_content, print_single_data
from verl.utils.toolplan.check_util import check_turns_data
import copy
import random
from verl.utils.toolplan.adv_compute import create_turns_and_cumulative_tensors
#role:system/user/[assistant/tool] , 理想状态assistant结尾
#content:
def get_api_names(search_oject):

    return f"{search_oject['category_name']}.{search_oject['tool_name']}.{search_oject['api_name']}"

def have_ground_truth_in_tool_response(search_response: list[dict], available_ground_truth_set):
    #print(search_response)
    api_names = [get_api_names(search_object) for search_object in search_response]
    api_name_set = set(api_names)
    #print("----------")
    #print(f"api_name_set:{api_name_set}")
    match_count = len(available_ground_truth_set & api_name_set)
    return match_count > 0, match_count, available_ground_truth_set & api_name_set

def cal_process_reward(single_data, ground_truth):

    ground_truth_list = copy.deepcopy(ground_truth)
    id_map = {s: idx for idx, s in enumerate(ground_truth_list)}
    available_ground_truth_set = set(ground_truth_list)
    ground_truth_len = len(ground_truth_list)
    #print_single_data(single_data)

    dialogue_view = _structure_single_dialogue(single_data)
    search_process_list = []
    for turn_view in dialogue_view.turns:
        if turn_view.role == 'system' or turn_view.role == 'user':
            continue
        elif turn_view.role == 'assistant':
            if turn_view.tool_calls:  # only has one tool call in here
                search_process_list.append(search_process(turn_view.tool_calls[0].arguments, [])) 
        elif turn_view.role == 'tool':
            search_process_list[-1].retrieval_api_names = parse_tools_from_retrieval_content(turn_view.content)
            # last search process maybe not get the retrieval content due to the limited length of context
            #obj = parse_tools_from_retrieval_content(turn_view.content)
            
    ids_per_turn = []
    event_turn = 0
    
    for seach_process in search_process_list:
        flag, match_count, match_api_names = have_ground_truth_in_tool_response(seach_process.retrieval_api_names, available_ground_truth_set)
        event_turn+=1
        if flag:
            available_ground_truth_set = available_ground_truth_set - match_api_names
            current_api_list = [id_map[api_name] for api_name in match_api_names]
            
            for i in range(event_turn-1):
                ids_per_turn.append([])
            ids_per_turn.append(current_api_list) #search_process_reward
            #reset event_turn to zero
            event_turn = 0
    
    for i in range(event_turn): # add zero reward for left wrong search process
        ids_per_turn.append([])

    turns_tensor, search_tensor = create_turns_and_cumulative_tensors(ground_truth_len,ids_per_turn)
    
    if dialogue_view.turns[-1].role == "assistant" and dialogue_view.turns[-1].tool_calls == None:
        has_answer_state = True
    else:
        has_answer_state = False
    
    # calulate the search_ratio
    ground_truth_set = set(ground_truth)
    apis_in_search_process = ground_truth_set-available_ground_truth_set
    search_ratio = len(ground_truth_set-available_ground_truth_set)*1.0 / len(ground_truth_set)
    #print(turns_tensor)
    return turns_tensor, search_tensor, has_answer_state, search_ratio, apis_in_search_process
            
    


@register("toolplan")
class ToolplanRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", reward_mode="gt_selection") -> None:
        """
        Initialize the ToolPlanRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.reward_mode = reward_mode

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32)
        turns_tensors = []
        search_tensors = []
        has_answer_states = []
        search_ratios = []
        selection_from_search_ratios=[]
        selection_from_gt_ratios=[]

        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            #print(data_item)
            
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            turns_tensor, search_tensor, has_answer_state, search_ratio, apis_in_search_process = cal_process_reward(data_item.non_tensor_batch['messages']['messages'], ground_truth)
            
           
            extra_info['apis_in_search_process'] = apis_in_search_process
            extra_info['reward_mode'] = self.reward_mode
            result_dict = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth, #考虑检索到的内容与ground_truth Recall and Precision to calcuate F1.
                extra_info=extra_info,
            )
            tool_selection_score = result_dict['tool_selection_score']
            selection_from_search_ratio = result_dict['selection_from_search_ratio']
            selection_from_gt_ratio = result_dict['selection_from_gt_ratio']

            if isinstance(tool_selection_score, dict):
                reward = tool_selection_score["score"]
                # Store the information including original reward
                for key, value in tool_selection_score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = tool_selection_score
                
            turns_tensors.append(turns_tensor)
            search_tensors.append(search_tensor)
            has_answer_states.append(has_answer_state)
            reward_tensor[i] = reward
            search_ratios.append(search_ratio)
            selection_from_search_ratios.append(selection_from_search_ratio)
            selection_from_gt_ratios.append(selection_from_gt_ratio)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            do_print = random.randint(1, 64) == 1
            if already_print_data_sources[data_source] < self.num_examine or do_print:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[turns_tensor]", turns_tensor)
                print("[search_tensor]", search_tensor)
                print("[has_answer_state]", has_answer_state)
                print("[search_ratio]", search_ratio)
                print("[selection_from_search_ratio]", selection_from_search_ratio)
                print("[selection_from_gt_ratio]", selection_from_gt_ratio)
                print("[apis_in_search_process]", apis_in_search_process)
                print("[tool_selection_score](have format)", tool_selection_score)
                print("[result_dict]", result_dict)
                score = tool_selection_score
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                #input()

        reward_extra_info['turns_tensors'] = turns_tensors
        reward_extra_info['search_tensors'] = search_tensors
        reward_extra_info['has_answer_states'] = has_answer_states
        reward_extra_info['search_ratios'] = search_ratios
        reward_extra_info['selection_from_search_ratios'] = selection_from_search_ratios
        reward_extra_info['selection_from_gt_ratios'] = selection_from_gt_ratios

        #search_ratio, selection_ratio_from_search, selection_ratio_from_groundtruth 
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
