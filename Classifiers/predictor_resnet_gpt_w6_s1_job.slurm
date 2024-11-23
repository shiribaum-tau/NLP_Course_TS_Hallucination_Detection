#! /bin/sh

#SBATCH --job-name=gpt_6_1_pred_resnet_job
#SBATCH --output=/home/joberant/NLP_2324b/ronimeamen/output/gpt_6_1_resnet_predictor.out # redirect stdout
#SBATCH --error=/home/joberant/NLP_2324b/ronimeamen/err/gpt_6_1_resnet_predictor.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --time=240 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --cpus-per-task=1 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


export XDG_CACHE_HOME="/home/joberant/NLP_2324b/ronimeamen/.cache"
export TORCH_HOME="/home/joberant/NLP_2324b/ronimeamen/.cache/torch_cache"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2324b/ronimeamen/.cache/transformers_cache"
export HF_HOME="/home/joberant/NLP_2324b/ronimeamen/.cache/hf_cache"
export CONDA_CACHE_DIR="/home/joberant/NLP_2324b/ronimeamen/.cache"
export PIP_CACHE_DIR="/home/joberant/NLP_2324b/ronimeamen/.cache"

echo 'Starting my SLURM job'

conda run -n pred_env python3 predictor_main.py --window_l 3 --stride 1 --batch_size 1024 --model ResNet --device gpu --data_gen_model openai --output_dir /home/joberant/NLP_2324b/ronimeamen/output --train_data_dir /home/joberant/NLP_2324b/ronimeamen/data/mini_GPT/Train --epochs 100 --saving_interval 20

echo 'Done'