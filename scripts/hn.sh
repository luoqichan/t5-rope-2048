#!/bin/bash
#SBATCH --job-name=cluster_negs
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=7-00:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

DATA_PATH="/data/user_data/luoqic/t5-rope-data"

split=documents
text_length=2048
n_gpus=4


train_qrels=$DATA_PATH/data/marco_documents_processed/qrels.train.tsv
train_queries=$DATA_PATH/data/marco_documents_processed/train.query.txt
corpus=$DATA_PATH/data/marco_documents_processed/corpus_firstp_2048.tsv


train_data_folder=$DATA_PATH/data/training_data/t5-rope-bkt-warmup

train_data=$train_data_folder/train.jsonl
valid_data=$train_data_folder/val.jsonl
model_path=/data/user_data/luoqic/t5-rope-data/models/t5-rope-2048-marco-bkt


# python OpenMatch/scripts/msmarco/build_hn.py  \
#     --tokenizer_name $model_path \
#     --hn_file $DATA_PATH/data/negatives/t5-rope-bkt-warmup/L1.CN.trec \
#     --qrels $train_qrels \
#     --queries $train_queries  \
#     --collection $corpus  \
#     --save_to $train_data_folder  \
#     --doc_template "Title: <title> Text: <text>" \
#     --n_sample 9 \
#     --truncate $text_length

python OpenMatch/scripts/msmarco/build_hn.py  \
    --tokenizer_name $model_path \
    --hn_file /data/user_data/luoqic/t5-rope-data/data/negatives/t5-rope-2048-marco-bkt/train.trec \
    --qrels $train_qrels \
    --queries $train_queries  \
    --collection $corpus  \
    --save_to $train_data_folder  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 9 \
    --truncate $text_length

cat $train_data_folder/*.hn.jsonl > $train_data_folder/full.jsonl
rm $train_data_folder/*.hn.jsonl

