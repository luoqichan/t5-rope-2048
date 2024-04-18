#!/bin/bash
#SBATCH --job-name=combine_neg
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=babel-4-28,shire-1-6,babel-2-12
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=8:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

hn_dir=/data/user_data/luoqic/t5-rope-data/data/training_data/t5-base-marco-documents-2048-self-hn-1
# cn_dir=/compute/shire-1-6/luoqic
cn_dir=/compute/babel-4-7/luoqic/t5-base-marco-documents-2048-bkt-hncn
save_dir=/compute/babel-4-7/luoqic/t5-rope-hncn-updated

mkdir -p $save_dir

python scripts/experiments/bkt_combine_negatives.py \
        --hard_negatives_dir $hn_dir \
        --cluster_negatives_dir $cn_dir \
        --save_dir $save_dir

