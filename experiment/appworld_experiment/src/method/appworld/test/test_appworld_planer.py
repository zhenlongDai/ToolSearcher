from src.method.appworld.appworld_planer import AppWorldPlaner
import argparse
from appworld import update_root
import os   
from src.utils.json_util import load_data_from_json, save_data_to_json
from appworld.task import Task, load_task_ids

def get_app_description(task: Task) -> str:
    description = ""

    for app_name, app_desc in task.app_descriptions.items():
        description += f"App Name: {app_name}\nDescription: {app_desc}\n\n"

    return description
    

def run_planer(planer, dataset_name):

    task_ids = load_task_ids(dataset_name)
    onetask = Task.load(task_ids[0], ground_truth_mode="full")
    app_descriptions = get_app_description(onetask)

    for task_id in task_ids:
        task = Task.load(task_id, ground_truth_mode="full")
        print(f"Processing Task ID: {task_id}, Instruction: {task.instruction}")
        generated_text, api_documentation_string, plan_prior_samples_retrieved_apis,instruction_predicted_apis = planer.generate_plan(is_sample_plans = True, instruction = task.instruction, app_descriptions = app_descriptions)
        #print("api_documentation_string:", api_documentation_string)
        #print("instruction_predicted_apis", instruction_predicted_apis)
        #print("Good set: not in instruction_predicted_apis but in plan prior retrieved apis:", set(plan_prior_samples_retrieved_apis) - set(instruction_predicted_apis))
        predict_apis_set = set(plan_prior_samples_retrieved_apis) | set(instruction_predicted_apis)
        #print("len of retrieved docs:", len(predict_apis_set))
        #print("ground truth required apis:", task.ground_truth.required_apis)
        #print("both in ground truth and Good set:", set(task.ground_truth.required_apis) & (set(plan_prior_samples_retrieved_apis) - set(instruction_predicted_apis)))
        #print("lack apis:", set(task.ground_truth.required_apis) - predict_apis_set)
        #print(f"Generated Plan for Task ID {task_id}:\n{generated_text}\n")
        #input("Press Enter to see generated plan...")

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
    run_planer(planer, args.dataset_name)