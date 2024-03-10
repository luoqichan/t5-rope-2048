import json
from tqdm import tqdm
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument('--negatives_dir', type=str, required=True)

args = parser.parse_args()
pdir = args.negatives_dir

print("Loading hard negatives...")
combined_negatives = {}
count = 0
with open(f"{pdir}/full.jsonl", "r") as hn:
    for i in tqdm(hn):
        line = json.loads(i)
        k = "-".join([str(x) for x in line["query"]])
        combined_negatives[k] = line

print("Loading and combining cluster_negatives...")
with open(f"{pdir}/full.cn.jsonl", "r") as cn:
    for i in tqdm(cn): 
        line = json.loads(i)
        k = "-".join([str(x) for x in line["query"]])
        combined_negatives[k]["cluster_negatives"] = line["negatives"]


print("Writing and saving to file...")
with open(f"{pdir}/full.combined.jsonl", "w") as wf:
    for _, dat in tqdm(combined_negatives.items()):
        if "cluster_negatives" in dat.keys():
            json.dump(dat, wf)
            wf.write("\n")



