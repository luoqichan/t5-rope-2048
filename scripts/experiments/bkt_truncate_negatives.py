import json
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--negatives_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)

args = parser.parse_args()
negatives_dir = args.negatives_dir
save_dir = args.save_dir

print("Truncating for val.jsonl ...")
with open(f"{negatives_dir}/val.jsonl", "r") as rf:
    with open(f"{save_dir}/val.jsonl", 'w') as wf:
        for i in tqdm(rf):
            line = json.loads(i)

            truncated = []
            for e, cluster in enumerate(line['cluster_negatives']):
                truncated.append(cluster[:e + 1])

            line['cluster_negatives'] = truncated
        
            json.dump(line, wf)
            wf.write("\n")

print("Truncating for train.jsonl ...")
with open(f"{negatives_dir}/train.jsonl", "r") as rf:
    with open(f"{save_dir}/train.jsonl", 'w') as wf:
        for i in tqdm(rf):
            line = json.loads(i)

            truncated = []
            for e, cluster in enumerate(line['cluster_negatives']):
                truncated.append(cluster[:e + 1])

            line['cluster_negatives'] = truncated
        
            json.dump(line, wf)
            wf.write("\n")
