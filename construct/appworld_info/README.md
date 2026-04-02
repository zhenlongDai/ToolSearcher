```bash
# Install and download:
mkdir /opt/conda/envs/app12
tar -xzf py12.tar.gz -C /opt/conda/envs/app12
#conda create -n app12 python==3.12
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install appworld
cd ./experiment/appworld_experiment/appworld
pip install -e .  
```