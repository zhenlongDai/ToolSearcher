# 让 Python/requests/httpx 用 curl 的 CA 文件
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

python -m src.selection_experiment.APIpredictor \
    --model_name gpt-4o-mini-ca \
    --dataset_mode traindev \
    --save_file_path "/ossfs/workspace/hy65/dzl/code/toolPlaner/experiment_results/appworld/APIpredictor/selection_gpt4o_mini_ca.json"

# python -m src.caluate_experiment.cal_retrieve_Recall \
#     --experiment_name unixcoder_retrieve_recall \
#     --model_name unixcoder \
#     --dataset_name dev 
#  "demo_task_ids": [
    #     "82e2fac_1",
    #     "29caf6f_1",
    #     "d0b1f43_1"
    # ],
