#! /bin/sh

#SBATCH --job-name=combine_job
#SBATCH --output=/home/joberant/NLP_2324b/ronimeamen/output/combine_data.out # redirect stdout
#SBATCH --error=/home/joberant/NLP_2324b/ronimeamen/err/combine_data.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --time=150 # max time (minutes)
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

conda run -n pred_env python3 create_combined_data_file.py --output_dir /home/joberant/NLP_2324b/ronimeamen/data/mini_GPT --logits_dir /home/joberant/NLP_2324b/ronimeamen/logits/mini_GPT --labels_dir /home/joberant/NLP_2324b/ronimeamen/labels/mini_GPT

echo 'Done'