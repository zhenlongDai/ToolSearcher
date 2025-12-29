import argparse
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
from copy import deepcopy
from typing import Any

from appworld import update_root
from appworld.common.background_server import BackgroundServer
from appworld.common.collections import override_dict
from appworld.common.io import write_json
from appworld.common.io import jsonnet_load
from appworld.common.path_store import path_store
from appworld.task import load_task_ids
from typing import Any, cast
from appworld.cli import evaluate
from appworld.task import Task, load_task_ids
from src.utils.agent import Agent
import src.method.appworld.full_code_agent
#import src.method.appworld.PlanExecRefl_agent
from src.utils.process_config import load_experiment_config

# def extract_dataset_name(runner_config: dict[str, Any]) -> str:
#     if "dataset" not in runner_config:
#         raise Exception("Dataset name not found in the runner config.")
#     return cast(str, runner_config["dataset"])


def run_experiment(
    experiment_name: str,
    runner_config: dict[str, Any],
    task_id: str | None = None,
    num_processes: int = 1,
    process_index: int = 0,
) -> None:
    agent_config = runner_config.pop("agent")
    print(runner_config)

    dataset_name = runner_config.pop("dataset")
    if runner_config:
        raise Exception(f"Unexpected keys in the runner config: {runner_config}")
    if task_id:
        task_ids = [task_id]
    else:
        task_ids = load_task_ids(dataset_name)
    # Done to assure all the tasks can be loaded fine without running any of them.
    for task_id in task_ids:
        Task.load(task_id=task_id)
    agent = Agent.from_dict(agent_config)
    print("experiment_name:", experiment_name)
    agent.solve_tasks(
        task_ids=task_ids,
        experiment_name=experiment_name,
        num_processes=num_processes,
        process_index=process_index,
    )


def run_experiment_cli(
    experiment_name: str,
    model_name: str = None,
    agent_name: str = None,
    dataset_name: str = None,
    task_id: str = None,
    with_evaluation: bool = False,
    num_processes: int = 1,
    process_index: int = None,
    root: str = ".",
    #experiment_outputs: str = "./outputs"
):
    update_root(root)
    experiment_config = load_experiment_config(
        model_name=model_name,
        agent_name=agent_name,
        dataset_name=dataset_name,
    )
    runner_config = experiment_config.pop("config")
    os.environ["MODEL_SERVER_URL"] = ""
    run_experiment(
        experiment_name=experiment_name,
        runner_config=deepcopy(runner_config),
        task_id=task_id,
        num_processes=num_processes,
        process_index=process_index,
    )
    if process_index is None and with_evaluation:
        if task_id is not None:
            dataset_name = None
        evaluate(
            experiment_name=experiment_name, dataset_name=dataset_name, task_id=task_id, root=root
        )


def main():
    parser = argparse.ArgumentParser(description="Run an AppWorld agent experiment.")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment to run.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--agent_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--task_id", type=str, default=None)
    parser.add_argument("--with_evaluation", action="store_true")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--process_index", type=int, default=None)
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    run_experiment_cli(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        agent_name=args.agent_name,
        dataset_name=args.dataset_name,
        task_id=args.task_id,
        with_evaluation=args.with_evaluation,
        num_processes=args.num_processes,
        process_index=args.process_index,
        root=args.root,
    )

if __name__ == "__main__":
    main()