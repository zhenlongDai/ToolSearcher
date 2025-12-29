from appworld.task import Task
from appworld import update_root
from src.utils.api_EmbeddingPredictor import APIEmbeddingPredictor
from src.utils.api_LLMpredictor import APILLMPredictor
from src.utils.process_config import  get_api_predictor_config
from appworld.task import Task, load_task_ids
from appworld import update_root
import argparse
import os
import json
from src.utils.json_util import load_data_from_json
from appworld import AppWorld
from appworld.environment import SAID_AVAILABLE_IMPORTS
from src.utils.common.utils import fill_model_server_url


def compute_recall(predicted_apis: list[str], ground_truth_apis: list[str]) -> float:
    # ({len(set(predicted_apis) & set(ground_truth_apis))}/{len(ground_truth_apis)})
    if not ground_truth_apis:
        raise ValueError("Ground truth APIs list is empty, cannot compute recall.")
    
    recall = len(set(predicted_apis) & set(ground_truth_apis)) / len(ground_truth_apis) if ground_truth_apis else 0.0
    return recall

def compute_precision(predicted_apis: list[str], ground_truth_apis: list[str]) -> float:
    if not predicted_apis:
        return 0.0
    precision = len(set(predicted_apis) & set(ground_truth_apis)) / len(predicted_apis)
    return precision

def compute_recall_for_appworld(task_ids: Task, predictor):
    total = 0
    recall_sum = 0
    precision_sum = 0
    #print(len(task_ids))
    #input("Press Enter to continue...")
    for task_id in task_ids:
        task = Task.load(task_id, ground_truth_mode="full")
        predicted_apis, _ = predictor.predict(task)
        #print(f"Predicted APIs for task {task_id}: {predicted_apis}")
        #input("Press Enter to continue...")
        # 假设 task.ground_truth_apis 是 list[str]
        ground_truth_apis = task.ground_truth.required_apis 
        #print(f"Ground Truth APIs for task {task_id}: {ground_truth_apis}")
        recall = compute_recall(predicted_apis, ground_truth_apis)
        precision = compute_precision(predicted_apis, ground_truth_apis)
        print(f"Task {task_id}:  recall={recall:.2f} precision={precision:.2f}")
        recall_sum += recall
        precision_sum += precision
        total += 1

    print(f"\nTotal tasks: {total}")
    print(f"Average recall: {recall_sum/total if total else 0:.4f}")
    print(f"Average precision: {precision_sum/total if total else 0:.4f}")

    pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AppWorld agent experiment.")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment to run.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--retrieve_mode", type=str, default="embedding", help="The retrieve mode: embedding or llm")
    parser.add_argument("--root", type=str, default="./appworld")
    args = parser.parse_args()
    update_root(args.root)
    
   
    if args.retrieve_mode == "embedding":
        model_config_path = os.path.join("./src/configs/retriever_config", f"{args.model_name}.json")
        model_config = load_data_from_json(model_config_path)
        print("model_config", model_config)
        
        predictor = APIEmbeddingPredictor(
            model_config=model_config,
            prompt_file_path="",
            demo_task_ids=[],
            max_predicted_apis= model_config['topk'],
            app_api_separator=".",
            mode="predicted",
        )
    elif args.retrieve_mode == "llm":
        api_predictor_config_path = os.path.join("./src/configs/api_predictor_config", f"appworld_LLMPredictor.json")
        api_predictor_config = get_api_predictor_config(
            model_name=args.model_name,
            api_predictor_config_path=api_predictor_config_path,
            prompt_file_path="",
        )
        print("apis number:", api_predictor_config['max_predicted_apis'])
        print(api_predictor_config)
        os.environ["MODEL_SERVER_URL"] = ""
        base_url = api_predictor_config["model_config"].get("base_url", None)
        if base_url:
            api_predictor_config["model_config"]["base_url"] = fill_model_server_url(base_url)
        predictor = APILLMPredictor(
            model_config=api_predictor_config['model_config'],
            prompt_file_path=api_predictor_config['prompt_file_path'],
            demo_task_ids=api_predictor_config['demo_task_ids'],
            max_predicted_apis= int(api_predictor_config['max_predicted_apis']),
            app_api_separator=".",
            mode="predicted",
            is_save_response=api_predictor_config['is_save_response'],
            save_predicted_apis_dir=api_predictor_config['save_predicted_apis_dir'],
        )
        
    
    task_ids = load_task_ids(args.dataset_name)
    compute_recall_for_appworld(task_ids, predictor)
    if args.retrieve_mode == "llm" and api_predictor_config['is_save_response']:
            predictor.save_predicted_results()