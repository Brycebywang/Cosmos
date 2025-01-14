## conda install

conda create -n "cosmos" python==3.10

pip install -r ../requirements.txt

## install megatron and nemo

pip install megatron_core

pip install git+https://github.com/NVIDIA/NeMo.git

pip install pytorch_lightning

pip install einops transformer_engine[pytorch] 