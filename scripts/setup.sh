export DEBIAN_FRONTEND=noninteractive

curl -f https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo apt-key add -



sudo apt-get update -y
sudo apt-get install software-properties-common -y

DEBIAN_FRONTEND=noninteractive sudo add-apt-repository ppa:deadsnakes/ppa -y


pip install -U "jax[tpu]"


pip install gemma datasets 

pip install git+https://github.com/google-research/kauldron

pip install git+https://github.com/huggingface/transformers

pip install --upgrade Jinja2

pip install -e .