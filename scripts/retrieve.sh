#!/bin/bash
#SBATCH --job-name=train_dr_openmatch
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=4:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

n_gpus=4

DATA_PATH="/data/user_data/luoqic/t5-rope-data/"
model_name="t5-base-marco-documents-2048-self-hn-1-self-hn-2"
embeddings_out="$DATA_PATH/data/embeddings/dev/$model_name"
output_path="$DATA_PATH/models/$trained_model_name"
model_to_eval="/data/user_data/luoqic/t5-rope-data/models/$model_name"
dev_queries="$DATA_PATH/data/marco_documents_processed/dev.query.txt"
dev_qrels="$DATA_PATH/data/marco_documents_processed/qrels.dev.tsv"
run_save="$DATA_PATH/results/$model_name"

mkdir -p $run_save
mkdir -p $embeddings_out

# already have corpus embeddings from hard negatives
cp $DATA_PATH/data/embeddings/train/$model_name/embeddings.corpus.* $DATA_PATH/data/embeddings/dev/$model_name

accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/retrieve.py  \
    --output_dir $embeddings_out  \
    --model_name_or_path $model_to_eval \
    --per_device_eval_batch_size 600  \
    --query_path $dev_queries \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $run_save/dev.trec \
    --dataloader_num_workers 1 \
    --use_gpu \
    --retrieve_depth 100 \
    --rope True

python OpenMatch/scripts/evaluate.py $dev_qrels $run_save/dev.trec > $run_save/dev.results