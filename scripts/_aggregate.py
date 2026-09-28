import json, glob

rows = []
for m in sorted(glob.glob("runs/*/metrics.json")):
    j = json.load(open(m))
    rows.append({
        "tag": j["tag"], "model": j["model"], "retr": j["retrieval"],
        "k": j["top_k"], "pool": j["pooling"],
        "acc": round(j["test_accuracy"]*100, 2),
        "f1": round(j["test_macro_f1"]*100, 2),
        "tr_s": round(j["train_samples_per_second"], 1),
        "tr_t": round(j.get("train_runtime", 0), 1),
        "te_s": round(j["test_samples_per_second"], 1),
    })

for r in sorted(rows, key=lambda r: -r["acc"]):
    print(f"{r['tag']:<26} {r['model']:<8} retr={r['retr']!s:<5} k={r['k']} pool={r['pool']:<5} "
          f"acc={r['acc']} f1={r['f1']}  train={r['tr_t']}s({r['tr_s']}/s)  test {r['te_s']}/s")