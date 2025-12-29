export CUDA_VISIBLE_DEVICES=0
python -m src.method.appworld.test.test_appworld_planer \
    --experiment_name test_appworld_planer \
    --agent_name appworld_planer \
    --dataset_name train 


