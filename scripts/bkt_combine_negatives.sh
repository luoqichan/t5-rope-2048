#!/bin/bash
#SBATCH --job-name=combine_engs
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=8:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

pdir=/data/user_data/luoqic/t5-rope-data/data/training_data/t5-rope-bkt-warmup-HN+CN-1
hn_file=full.jsonl
cn_file=full.cn.jsonl
save_file=full.hncn.jsonl

python scripts/experiments/bkt_combine_negatives.py --negatives_dir $pdir

