from src.method.appworld.appworld_planer import AppWorldPlaner
import argparse
from appworld import update_root
import os   
from src.utils.json_util import load_data_from_json, save_data_to_json
from appworld.task import Task, load_task_ids
from tqdm import tqdm

def get_app_description(task: Task) -> str:
    description = ""

    for app_name, app_desc in task.app_descriptions.items():
        description += f"App Name: {app_name}\nDescription: {app_desc}\n\n"

    return description
    
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
        
def compute_recall_for_appworld(planer, dataset_name):

    task_ids = load_task_ids(dataset_name)
    onetask = Task.load(task_ids[0], ground_truth_mode="full")
    app_descriptions = get_app_description(onetask)
    total = 0
    recall_sum = 0
    precision_sum = 0
    same_recall_sum = 0
    same_precision_sum = 0
    for task_id in tqdm(task_ids):
        task = Task.load(task_id, ground_truth_mode="full")
        print(f"Processing Task ID: {task_id}, Instruction: {task.instruction}")
        generated_text, api_documentation_string, plan_prior_samples_retrieved_apis,instruction_predicted_apis = planer.generate_plan(is_sample_plans = True, instruction = task.instruction, app_descriptions = app_descriptions)
       
        predict_apis_set = set(plan_prior_samples_retrieved_apis) | set(instruction_predicted_apis)
        ground_truth_apis = task.ground_truth.required_apis 
        predict_apis_list = list(predict_apis_set)
        recall = compute_recall(predict_apis_list, ground_truth_apis)
        precision = compute_precision(predict_apis_list, ground_truth_apis)
        same_count_predict_apis = planer.predictor.predict(task, topk=len(predict_apis_list))[0]
        print("same_count_predict_apis:", len(same_count_predict_apis),"predict_apis_list:", len(predict_apis_list) )


        # print("api_documentation_string:", api_documentation_string)
        # print("instruction_predicted_apis", instruction_predicted_apis)
        # print("len of retrieved docs:", len(predict_apis_set))
        # print("ground truth required apis:", task.ground_truth.required_apis)
        # print(f"Ground Truth APIs for task {task_id}: {ground_truth_apis}")
        
        recall_sum += recall
        precision_sum += precision
        total += 1       
        same_recall = compute_recall(same_count_predict_apis, ground_truth_apis)
        same_precision = compute_precision(same_count_predict_apis, ground_truth_apis)
        same_recall_sum += same_recall
        same_precision_sum += same_precision
        print(f"Task {task_id}:  recall={recall:.2f} precision={precision:.2f},")
        print(f"Task {task_id} with same count predicted APIs:  recall={same_recall:.2f} precision={same_precision:.2f}")
        #input("Press Enter to continue...")
        #input("Press Enter to continue...")

    print(f"\nTotal tasks: {total}")
    print(f"Average recall: {recall_sum/total if total else 0:.4f}")
    print(f"Average precision: {precision_sum/total if total else 0:.4f}")
    print(f"Average same count recall: {same_recall_sum/total if total else 0:.4f}")
    print(f"Average same count precision: {same_precision_sum/total if total else 0:.4f}")
                                           

def save_all_apis(task: Task, file_path: str):
    api_docs_json = {}
    for app_name, api_docs in task.api_docs.items():
        api_docs_json[app_name] = api_docs
    save_data_to_json(api_docs_json, file_path)
    print(f"All API docs saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AppWorld agent experiment.")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment to run.")
    parser.add_argument("--agent_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--root", type=str, default="./appworld")
    args = parser.parse_args()
    update_root(args.root)
    model_config_path = os.path.join("./src/configs/agent_config", f"{args.agent_name}.json")
    model_config = load_data_from_json(model_config_path)
    print("model_config", model_config)
    #save_all_apis(Task.load("82e2fac_1", load_ground_truth=False), "./all_api_docs.json")
    planer = AppWorldPlaner(**model_config)
    compute_recall_for_appworld(planer, args.dataset_name)