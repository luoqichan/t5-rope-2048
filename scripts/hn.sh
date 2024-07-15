#!/bin/bash
#SBATCH --job-name=build_hn
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

DATA_PATH="/data/user_data/luoqic/t5-rope-data"

split=documents
text_length=2048


train_qrels=$DATA_PATH/data/marco_documents_processed/qrels.train.tsv
train_queries=$DATA_PATH/data/marco_documents_processed/train.query.txt
corpus=$DATA_PATH/data/marco_documents_processed/corpus_firstp_2048.tsv

train_data_folder=$DATA_PATH/data/training_data/t5-base-marco-documents-2048-bkt_hncn_ranked_intersect
model_path=/data/user_data/luoqic/t5-rope-data/models/t5-base-marco-documents-2048

train_data_folder=$DATA_PATH/data/training_data/$trained_model_name
train_data=$train_data_folder/train.jsonl
valid_data=$train_data_folder/val.jsonl
run_save=$DATA_PATH/data/negatives/$trained_model_name

python OpenMatch/scripts/msmarco/build_hn.py  \
    --tokenizer_name $output_path \
    --hn_file /data/user_data/luoqic/bkt-cluster/experiments/intersect/combined_ranked.txt \
    --qrels $train_qrels \
    --queries $train_queries  \
    --collection $corpus  \
    --save_to $train_data_folder  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 9 \
    --truncate $text_length

cat $train_data_folder/*.hn.jsonl > $train_data_folder/full.jsonl
rm $train_data_folder/*.hn.jsonl

line_count=$(wc -l $train_data_folder/full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $train_data_folder/full.jsonl > $train_data_folder/val.jsonl
head -n $n_train $train_data_folder/full.jsonl > $train_data_folder/train.jsonl

rm $train_data_folder/full.jsonl

