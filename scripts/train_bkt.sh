#!/bin/bash
#SBATCH --job-name=CN_debug
#SBATCH --exclude=babel-4-28,babel-3-19
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=7-00:00:00

eval "$(conda shell.bash hook)"
conda activate openmatch
export PYTHONPATH=/home/luoqic/t5-rope-2048
export WANDB_PROJECT=t5-rope-bkt-hncn-separate-loss

mkdir -p /scratch/luoqic

split=documents
text_length=2048
n_gpus=4
DATA_PATH=/data/user_data/luoqic/t5-rope-data
train_qrels=$DATA_PATH/data/marco_documents_processed/qrels.train.tsv
train_queries=$DATA_PATH/data/marco_documents_processed/train.query.txt
corpus=$DATA_PATH/data/marco_documents_processed/corpus_firstp_2048.tsv

initial_model=$DATA_PATH/models/t5-base-marco-documents-2048
train_data_folder=$DATA_PATH/data/training_data/t5-base-marco-documents-2048-bkt
# train_data=$train_data_folder/train.jsonl
# valid_data=$train_data_folder/val.jsonl
train_data=/compute/shire-1-6/luoqic/train.jsonl
valid_data=/compute/shire-1-6/luoqic/val.jsonl 

trained_model_name=t5-base-marco-documents-2048-HNCN-separatelosses-debug-eval
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

# embeddings_out=$DATA_PATH/data/embeddings/train/$trained_model_name
# run_save=$DATA_PATH/data/negatives/$trained_model_name

# mkdir -p $run_save
# mkdir -p $embeddings_out

# accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/build_index.py  \
#     --output_dir $embeddings_out \
#     --model_name_or_path $output_path  \
#     --per_device_eval_batch_size 430 \
#     --corpus_path $corpus  \
#     --doc_template "Title: <title> Text: <text>"  \
#     --doc_column_names id,title,text  \
#     --p_max_len $text_length  \
#     --fp16  \
#     --dataloader_num_workers 1

# accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/retrieve.py  \
#     --output_dir $embeddings_out  \
#     --model_name_or_path $output_path  \
#     --per_device_eval_batch_size 600  \
#     --query_path $train_queries  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 32  \
#     --fp16  \
#     --trec_save_path $run_save/train.trec \
#     --dataloader_num_workers 1 \
#     --use_gpu

# model_to_eval=$output_path
# model_name=$(basename "$model_to_eval")

# embeddings_out="$DATA_PATH/data/embeddings/dev/$model_name"
# run_save="$DATA_PATH/results/$model_name"

# dev_queries=$DATA_PATH/data/marco_documents_processed/dev.query.txt
# dev_qrels=$DATA_PATH/data/marco_documents_processed/qrels.dev.tsv

# mkdir -p $run_save
# mkdir -p $embeddings_out

# # already have corpus embeddings from hard negatives
# cp $DATA_PATH/data/embeddings/train/$model_name/embeddings.corpus.* $DATA_PATH/data/embeddings/dev/$model_name

# accelerate launch --num_processes $n_gpus --multi_gpu --main_process_port 29777 OpenMatch/src/openmatch/driver/retrieve.py  \
#     --output_dir $embeddings_out  \
#     --model_name_or_path $model_to_eval \
#     --per_device_eval_batch_size 600  \
#     --query_path $dev_queries \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 32  \
#     --fp16  \
#     --trec_save_path $run_save/dev.trec \
#     --dataloader_num_workers 1 \
#     --use_gpu \
#     --retrieve_depth 100 \
#     --rope True

# python OpenMatch/scripts/evaluate.py $dev_qrels $run_save/dev.trec > $run_save/dev.results

# rm -r $DATA_PATH/data/embeddings/dev/$model_name/embeddings.corpus.*