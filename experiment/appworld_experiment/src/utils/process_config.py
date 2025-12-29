import json
import os

def load_experiment_config(
    model_name: str | None = None,
    agent_name: str | None = None,
    dataset_name: str | None = None,
) -> dict:
    with open(os.path.join("./src/configs/agent_config", agent_name + ".json"), "r") as f:
        experiment_config = json.load(f)
    
    if model_name is not None:
        with open(os.path.join("./src/configs/model_config/model.json"), "r") as f:
            all_model_config = json.load(f)
        model_config = all_model_config.get(model_name)
        if model_config is None:
            raise Exception(f"Model config for model '{model_name}' not found.")
        experiment_config["config"]["agent"]["api_predictor_config"]["model_config"] = model_config
        experiment_config["config"]["agent"]["model_config"] = model_config
    if dataset_name is not None:
        experiment_config["config"]["dataset"] = dataset_name
    return experiment_config

def set_api_key_by_model_config(
    model_config: dict,
) -> dict:
    with open(os.path.join("./src/configs/key.json"), "r") as f:
        key_config = json.load(f)

    api_key_env_name = model_config["api_key_env_name"]
    api_key_value = key_config.get(api_key_env_name)
    if api_key_value is None:
        raise Exception(f"API key for '{api_key_env_name}' not found in key.json.")
    os.environ[api_key_env_name] = api_key_value
    print(f"Set API key for '{api_key_env_name}':'{api_key_value}' successfully.")
    

def get_api_predictor_config(
    model_name: str,
    api_predictor_config_path: str,
    prompt_file_path: str = "",
) -> dict:
    with open(os.path.join("./src/configs/model_config/model.json"), "r") as f:
        all_model_config = json.load(f)
    model_config = all_model_config.get(model_name)
    with open(api_predictor_config_path, "r") as f:
        experiment_config = json.load(f)
    experiment_config["model_config"] = model_config
    
    if prompt_file_path != "":
        experiment_config["prompt_file_path"] = prompt_file_path

    return experiment_config