import json
from tqdm import tqdm
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument('--negatives_dir', type=str, required=True)
parser.add_argument('--hn_file', type=str, required=True)
parser.add_argument('--cn_file', type=str, required=True)
parser.add_argument('--save_file', type=str, required=True)

args = parser.parse_args()
pdir = args.negatives_dir

print("Loading hard negatives...")
combined_negatives = {}
count = 0
with open(f"{pdir}/{args.hn_file}", "r") as hn:
    for i in tqdm(hn):
        line = json.loads(i)
        k = "-".join([str(x) for x in line["query"]])
        combined_negatives[k] = line

print("Loading and combining cluster_negatives...")
with open(f"{pdir}/{args.cn_file}", "r") as cn:
    for i in tqdm(cn): 
        line = json.loads(i)
        k = "-".join([str(x) for x in line["query"]])
        # combined_negatives[k]["negatives"] = combined_negatives[k]["negatives"][:5]
        # combined_negatives[k]["negatives"] += line["negatives"][:4]

        combined_negatives[k]["cluster_negatives"] = line["negatives"]


print("Writing and saving to file...")
with open(f"{pdir}/{args.save_file}", "w") as wf:
    for _, dat in tqdm(combined_negatives.items()):
        json.dump(dat, wf)
        wf.write("\n")



