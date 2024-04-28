#!/bin/bash
#SBATCH --job-name=combine_engs
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=babel-4-28,shire-1-6,babel-2-12
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=8:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

negatives_dir=/compute/babel-4-7/luoqic/t5-rope-hncn-updated
save_dir=/compute/shire-1-6/luoqic/t5-rope-hncn-truncated

mkdir -p $save_dir

python scripts/experiments/bkt_truncate_negatives.py \
        --negatives_dir $negatives_dir \
        --save_dir $save_dir

