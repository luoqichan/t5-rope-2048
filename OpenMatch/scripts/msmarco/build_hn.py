# Adapted from Tevatron (https://github.com/texttron/tevatron)

from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from openmatch.utils import SimpleTrainPreProcessor as TrainPreProcessor


def load_ranking(rank_file, relevance, n_sample, depth, split_token):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, _, p_0, _, _, _ = next(lines).strip().split()

        curr_q = q_0
        negatives = [] if p_0 in relevance[q_0] else [p_0]

        while True:
            try:
                q, _, p, _, _, _ = next(lines).strip().split()
                if q != curr_q:
                    negatives = negatives[:depth]
                    random.shuffle(negatives)
                    yield curr_q, relevance[curr_q], negatives[:n_sample], split_token
                    curr_q = q
                    negatives = [] if p in relevance[q] else [p]
                else:
                    if p not in relevance[q]:
                        negatives.append(p)
            except StopIteration:
                negatives = negatives[:depth]
                random.shuffle(negatives)
                yield curr_q, relevance[curr_q], negatives[:n_sample], split_token
                return

def sample_across_clusters(negatives:dict, n_samples:int):

    sample_distr = {5: [2, 2, 2, 2, 1], 4:[3, 2, 2, 2], 3: [3, 3, 3], 2:[5, 4], 1:[9]}

    sampled_negatives = []
    keys = list(negatives.keys())
    num_levels = len(keys)

    gold = False
    if 'gold_leaf' in keys:
        keys.remove('gold_leaf')
        gold = True
    keys = sorted(keys, reverse=True)
    if gold: 
        keys.append('gold_leaf')

    overflow = 0

    distr = sample_distr[num_levels]
    unsampled_pool = []

    for enum, k in enumerate(keys): 
        num_to_sample = distr[enum] + overflow
        overflow = 0

        if len(negatives[k]) < num_to_sample:
            sampled_negatives += negatives[k]
            overflow += (int(distr[enum]) - len(negatives[k]))

        else:
            random.shuffle(negatives[k])
            sampled_negatives += (negatives[k][:num_to_sample])

            unsampled_pool += negatives[k][num_to_sample:]

    if len(sampled_negatives) < n_samples: 
        
        needed = n_samples - len(sampled_negatives)
        random.shuffle(unsampled_pool)
        sampled_negatives += unsampled_pool[:needed]

    assert len(sampled_negatives) == n_samples
    return sampled_negatives 

def load_ranking_with_cluster_distr(rank_file, relevance, n_sample, depth, split_token):
    with open(rank_file) as rf:
        neg_count = 0
        lines = iter(rf)
        q_0, p_0, _, level_0 = next(lines).strip().split()

        curr_q = q_0
        
        if p_0 in relevance[q_0]:
            negatives = {}
        else: 
            negatives = {}
            negatives[level_0] = [p_0]
            neg_count += 1

        while True:
            try:
                q, p, _, level = next(lines).strip().split()

                if neg_count == depth:
                    # select negatives from limited top-n! 

                    sampled_negatives = sample_across_clusters(negatives, n_sample)
                    neg_count = -1000
                    yield curr_q, relevance[curr_q], sampled_negatives, split_token


                if q != curr_q:
                    # reset for next query
                    neg_count = 0
                    curr_q = q
                    if p_0 in relevance[q_0]:
                        negatives = {}
                    else: 
                        negatives[level] = [p_0]
                        neg_count += 1

                else:
                    if p not in relevance[q]:
                        if level not in negatives: 
                            negatives[level] = [p]
                        else: 
                            negatives[level].append(p)
                        neg_count += 1
                            
            except StopIteration:
                sampled_negatives = sample_across_clusters(negatives, n_sample)
                yield curr_q, relevance[curr_q], sampled_negatives, split_token
                return


random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--hn_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--doc_template', type=str, default=None)
parser.add_argument('--query_template', type=str, default=None)
parser.add_argument('--columns', type=str, default="text_id,title,text")


parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--truncate_q', type=int, default=32)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=45000)
parser.add_argument('--split_sentences', type=str, default=None)

args = parser.parse_args()

qrel = TrainPreProcessor.read_qrel(args.qrels)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    doc_max_len=args.truncate,
    query_max_len=args.truncate_q,
    doc_template=args.doc_template,
    query_template=args.query_template,
    allow_not_found=True,
    columns=args.columns.split(",")
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

# pbar = tqdm(load_ranking(args.hn_file, qrel, args.n_sample, args.depth, args.split_sentences))
pbar = tqdm(load_ranking_with_cluster_distr(args.hn_file, qrel, args.n_sample, args.depth, args.split_sentences))

with Pool() as p:
    for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
        counter += 1
        if f is None:
            f = open(os.path.join(args.save_to, f'split{shard_id:02d}.hn.jsonl'), 'w')
            pbar.set_description(f'split - {shard_id:02d}')
        f.write(x + '\n')

        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0

if f is not None:
    f.close()