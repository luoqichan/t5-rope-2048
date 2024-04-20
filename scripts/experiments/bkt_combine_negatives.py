import json
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--hard_negatives_dir', type=str, required=True)
parser.add_argument('--cluster_negatives_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)

args = parser.parse_args()
hn_dir = args.hard_negatives_dir
cn_dir = args.cluster_negatives_dir
save_dir = args.save_dir


print("Loading cluster negatives...")
hn_lookup = {} # {query_tokens: {record}} 
with open(f"{hn_dir}/train.jsonl", "r") as hn:
    for i in tqdm(hn):
        line = json.loads(i)
        k = "-".join([str(x) for x in line["query"]])
        hn_lookup[k] = line

with open(f"{hn_dir}/val.jsonl", "r") as hn:
    for i in tqdm(hn):
        line = json.loads(i)
        k = "-".join([str(x) for x in line["query"]])
        hn_lookup[k] = line

print("Loading and combining negatives...")

val_written = 0
train_written = 0
with open(f"{save_dir}/val.jsonl", "w") as wf:
    with open(f"{cn_dir}/val.jsonl", "r") as cn:
        for i in tqdm(cn): 
            line = json.loads(i)
            k = "-".join([str(x) for x in line["query"]])

            record = hn_lookup[k]
            record["cluster_negatives"] = line["cluster_negatives"]

            json.dump(record, wf)
            wf.write("\n")

            removed = hn_lookup.pop(k)
            val_written += 1

with open(f"{save_dir}/train.jsonl", "w") as wf:
    with open(f"{cn_dir}/train.jsonl", "r") as cn:
        for i in tqdm(cn): 
            line = json.loads(i)
            k = "-".join([str(x) for x in line["query"]])

            try:
                record = hn_lookup[k]
                record["cluster_negatives"] = line["cluster_negatives"]



                json.dump(record, wf)
                wf.write("\n")

                removed = hn_lookup.pop(k)
                train_written += 1
            except: 
                print(f"{k} not in hn_lookup")

print(f"{val_written} lines written to val.jsonl\n{train_written} lines written to train.jsonl")




