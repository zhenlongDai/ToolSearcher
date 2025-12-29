export CUDA_VISIBLE_DEVICES=7
python -m src.caluate_experiment.cal_plan_prior \
    --experiment_name test_appworld_planer \
    --agent_name appworld_planer \
    --dataset_name train 
