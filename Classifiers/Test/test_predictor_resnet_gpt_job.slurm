#! /bin/sh

#SBATCH --job-name=resnet_gpt_test_job
#SBATCH --output=/home/joberant/NLP_2324b/ronimeamen/output/resnet_gpt_test_predictor.out # redirect stdout
#SBATCH --error=/home/joberant/NLP_2324b/ronimeamen/err/resnet_gpt_test_predictor.err # redirect stderr
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

conda run -n pred_env python3 test_model.py --data_gen_model openai --model_name ResNet --model_file /home/joberant/NLP_2324b/ronimeamen/output/best_models/resnet_best_model_k=20_w=6_s=1.keras --output_dir /home/joberant/NLP_2324b/ronimeamen/output/test_models --test_data_dir /home/joberant/NLP_2324b/ronimeamen/data/mini_GPT/Test

echo 'Done'