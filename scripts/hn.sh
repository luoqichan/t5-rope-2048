#!/bin/bash
#SBATCH --job-name=hn
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=8:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

DATA_PATH="/data/user_data/luoqic/t5-rope-data"
text_length=2048

train_qrels=$DATA_PATH/data/marco_documents_processed/qrels.train.tsv
train_queries=$DATA_PATH/data/marco_documents_processed/train.query.txt
corpus=$DATA_PATH/data/marco_documents_processed/corpus_firstp_2048.tsv
negatives=$DATA_PATH/data/marco_documents_processed/train.negatives.tsv

trained_model_name=t5-rope-bkt-warmup-HN+CN-1

trec=$DATA_PATH/data/negatives/$trained_model_name
train_data_folder=$DATA_PATH/data/training_data/$trained_model_name
output_path=/data/user_data/luoqic/t5-rope-data/models/$trained_model_name

mkdir -p $train_data_folder

python OpenMatch/scripts/msmarco/build_hn.py  \
    --tokenizer_name $output_path \
    --hn_file $trec/train.trec \
    --qrels $train_qrels \
    --queries $train_queries  \
    --collection $corpus  \
    --save_to $train_data_folder  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 5 \
    --truncate $text_length

cat $train_data_folder/*.hn.jsonl > $train_data_folder/full.jsonl
rm $train_data_folder/*.hn.jsonl

line_count=$(wc -l $train_data_folder/full.jsonl | awk '{print $1}')

