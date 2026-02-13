from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig, parse_args, generate_config_from_args 



if __name__ == "__main__":
    args = parse_args()
    config = generate_config_from_args(args)
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
