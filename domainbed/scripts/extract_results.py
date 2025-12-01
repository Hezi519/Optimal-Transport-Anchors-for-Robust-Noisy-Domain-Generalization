import json, glob, re
import numpy as np
import pandas as pd
import os
os.chdir("results_submission")

pattern = re.compile(r"env\d+_out_acc")
rows = []

def parse_hparams(hp):
    if isinstance(hp, dict):
        return hp
    if isinstance(hp, str):
        try:
            return json.loads(hp)
        except Exception:
            return {}
    return {}

def ot_label(base_alg, ot_unbalanced):
    if base_alg != "OT":
        return base_alg
    if ot_unbalanced in ([1,1], 1, [1]):
        return "UOT"
    if ot_unbalanced in ([0,0], 0, [0], None):
        return "EOT"
    return f"OT({ot_unbalanced})"

for path in glob.glob("*.jsonl"):
    best = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            env_outs = {k: v for k, v in rec.items() if pattern.fullmatch(k)}
            if not env_outs:
                continue
            overall = float(np.mean(list(env_outs.values())))
            if best is None or overall > best["overall"]:
                best = {"rec": rec, "env_outs": env_outs, "overall": overall}
    if best is None:
        continue
    env_vals = np.array(list(best["env_outs"].values()), dtype=float)
    args = best["rec"].get("args", {})
    hparams = parse_hparams(args.get("hparams"))
    alg_label = ot_label(args.get("algorithm"), hparams.get("ot_unbalanced"))
    rows.append({
        "algorithm": alg_label,
        "dataset": args.get("dataset"),
        "mean_out_acc": float(env_vals.mean()),
        "best_step": best["rec"].get("step"),
        "source_file": path,
    })

best_df = pd.DataFrame(rows)
best_unique = (best_df.sort_values("mean_out_acc", ascending=False)
                        .drop_duplicates(subset=["algorithm","dataset"], keep="first"))
pivot_mean = best_unique.pivot(index="algorithm", columns="dataset",
                               values="mean_out_acc").sort_index()
pivot_2dp = pivot_mean.round(3)
print(pivot_2dp)
