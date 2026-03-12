### env install
#### env for training
```bash
conda create -n verl python==3.10
git checkout v0.5.0
pip install -e .
pip install vllm==0.8.2
pip install tensordict==0.6.2
pip install "sglang[all]==0.4.6.post5"
#pip install "sglang[all]>=0.4.5.post3"
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install ray==2.44.0
pip install ransformers==4.52.4
pip install /XXX/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install wandb
pip install munch
pip install scikit-learn
```
#### environment for retriever
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

pip install sentence_transformers==5.2.0
## API function
pip install uvicorn fastapi
pip install munch
```

```bash
conda create -n retriever_v2 python=3.10
conda activate retriever_v2

# we recommend installing torch with conda for faiss-gpu
#conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia


conda install pytorch==2.6.0 -c pytorch
pip install transformers datasets pyserini
#pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

pip install sentence_transformers==5.2.0
## API function
pip install uvicorn fastapi
pip install munch==4.0.0
```

#### Inference

##### Stage I: generate the content of the retrieval apis and plan
1. preprocess test data to parquet files
```bash 
bash ./scripts/construct/construct_stabletoolbench_test_data.sh
``` 