#! /bin/sh

#SBATCH --job-name=pythia_test
#SBATCH --output=/home/joberant/NLP_2324b/ronimeamen/output/pythia.out # redirect stdout
#SBATCH --error=/home/joberant/NLP_2324b/ronimeamen/err/pythia.err # redirect stderr
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

echo 'Starting my SLURM job'

conda run -n nlp_proj python3 main.py --device gpu --output_dir /home/joberant/NLP_2324b/ronimeamen/output --deterministic

echo 'Done'