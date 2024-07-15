#!/bin/bash
#SBATCH --job-name=eval_dr_openmatch
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:15:00

eval "$(conda shell.bash hook)"
conda activate openmatch

DATA_PATH="/data/user_data/luoqic/t5-rope-data"
dev_qrels=$DATA_PATH/data/marco_documents_processed/qrels.dev.tsv
run_save=/data/user_data/luoqic/bkt-cluster/experiments/t5-base-marco-documents-2048-bkt_hncn_intercept

python OpenMatch/scripts/evaluate.py $dev_qrels $run_save/dev.trec > $run_save/dev.results
