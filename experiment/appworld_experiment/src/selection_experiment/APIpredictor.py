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
from src.utils.json_util import load_data_from_json, save_data_to_json
from appworld import AppWorld
from appworld.environment import SAID_AVAILABLE_IMPORTS
from src.utils.common.utils import fill_model_server_url

from tqdm import tqdm

def tool_selection_for_appworld(dataset_names, predictor, save_file_path):
    data_list = []
    
    for dataset_name in dataset_names:
        task_ids = load_task_ids(dataset_name)
        mode = "full" if dataset_name in ["train", "dev"] else "minimal"
        for task_id in tqdm(task_ids):
            task = Task.load(task_id, ground_truth_mode=mode)
            predicted_apis, _ = predictor.predict(task)
            data_item = {
                "index": task_id,
                "data_source": dataset_name,
                "selected_apis": predicted_apis,
            }
            data_list.append(data_item)

    save_data_to_json(data_list, save_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AppWorld agent experiment.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_mode", type=str, default="traindev", help="traindev/test_normal/test_challenge")
    parser.add_argument("--root", type=str, default="./appworld")
    parser.add_argument("--save_file_path", type=str, default="traindev", help="traindev/test_normal/test_challenge")
    args = parser.parse_args()
    update_root(args.root)
    
    api_predictor_config_path = os.path.join("./src/configs/api_predictor_config", f"appworld_LLMPredictor.json")
    api_predictor_config = get_api_predictor_config(
        model_name=args.model_name,
        api_predictor_config_path=api_predictor_config_path,
        prompt_file_path="",
    )
    print("apis number:", api_predictor_config['max_predicted_apis'])

    os.environ["MODEL_SERVER_URL"] = ""
    base_url = api_predictor_config["model_config"].get("base_url", None)
    if base_url:
        api_predictor_config["model_config"]["base_url"] = fill_model_server_url(base_url)
    
    print(f"api_predictor_config:{api_predictor_config}")
    
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

    if args.dataset_mode == "traindev":
        dataset_names = ["train", "dev"]
    elif args.dataset_mode == "test_normal":
        dataset_names = ["test_normal"]
    elif args.dataset_mode == "test_challenge":
        dataset_names = ["test_challenge"]       

    tool_selection_for_appworld(dataset_names, predictor, args.save_file_path)
  