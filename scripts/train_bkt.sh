#!/bin/bash
#SBATCH --job-name=train_dr_openmatch
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --exclude=babel-4-28
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=7-00:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch

mkdir -p /scratch/luoqic

DATA_PATH="/data/user_data/luoqic/t5-rope-data"
export WANDB_PROJECT=t5-rope-bkt-hncn-separate-loss


split=documents
text_length=2048
n_gpus=4

num_episodes=4

train_qrels=$DATA_PATH/data/marco_documents_processed/qrels.train.tsv
train_queries=$DATA_PATH/data/marco_documents_processed/train.query.txt
corpus=$DATA_PATH/data/marco_documents_processed/corpus_firstp_2048.tsv
negatives=$DATA_PATH/data/marco_documents_processed/train.negatives.tsv

initial_model=$DATA_PATH/models/t5-base-marco-documents-2048
trained_model_name=t5-base-marco-$split-$text_length-self-hn-1-staticdata

train_data=/compute/shire-1-6/luoqic/train.jsonl
valid_data=/compute/shire-1-6/luoqic/val.jsonl 

output_path=$DATA_PATH/models/$trained_model_name

accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/train_dr.py  \
    --output_dir $output_path \
    --model_name_or_path $initial_model \
    --do_train \
    --save_steps 125  \
    --eval_steps 125  \
    --fp16 \
    --train_path $train_data  \
    --eval_path $valid_data  \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 4 \
    --train_n_passages 10  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len $text_length  \
    --num_train_epochs 2  \
    --report_to wandb \
    --logging_steps 10 \
    --run_name $trained_model_name \
    --evaluation_strategy steps \
    --dataloader_num_workers 4 \
    --rope True \
    --grad_cache True \
    --use_mapping_dataset True \
    --gc_p_chunk_size 24 \
    --gc_q_chunk_size 24 \
    --negatives_x_device True \
    --data_cache_dir /scratch/luoqic
