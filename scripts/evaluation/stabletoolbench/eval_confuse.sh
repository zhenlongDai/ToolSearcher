export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -m evaluation.eval_confus_score.eval_confuse_score \
  --predict_file_path "/ossfs/workspace/hy65/dzl/code/toolPlaner/experiment_results/toolsearcher/toolsearcher_ablation_top5.json" \
  --retrieved_file_path "/ossfs/workspace/hy65/dzl/code/toolPlaner/baseline_method/RAG/data/stabletoolbench_dataset/tool_selection_top100.parquet"\
  --toolbench_tools_dir "./experiment/stabletoolbench_experiment/StableToolBench/server/tools"\
  --save_vector_file_path "./evaluation/eval_confus_score/stabletoolbench/Qwen_vector_store.pkl"\
  --groundtruth_file_path "./data/stabletoolbench_dataset/tool_selection.parquet" \
  