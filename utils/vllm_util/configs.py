import argparse
import yaml
from dataclasses import dataclass, fields
from typing import Any, Dict

@dataclass
class InferenceConfig:

    model_path: str
    temperature: int
    top_p: int
    gpu_memory_utilization: float
    dtype: str
    max_input_tokens: int
    max_output_tokens: int
    dp_size: int
    tp_size: int
    dp_master_ip: str
    dp_master_port: int
    data_file_path: str
    debug_mode: bool
    save_file_path: str
    system_prompt: str
    use_lora: bool = False
    lora_path: str = ""
    extra: Dict[str, Any] = None   # 用来装多余字段
    

def load_config(config_path: str) -> InferenceConfig:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file) or {}
    field_names = {f.name for f in fields(InferenceConfig)}
    main_data = {k: v for k, v in config_data.items() if k in field_names}
    extra_data = {k: v for k, v in config_data.items() if k not in field_names}
    # 把多余字段塞进 extra
    main_data['extra'] = extra_data or {}
    return InferenceConfig(**main_data)