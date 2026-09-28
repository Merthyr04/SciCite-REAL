import json, sys, glob, re
dirpath = sys.argv[1] if len(sys.argv) > 1 else r"runs\real_deberta_dual_k3"
cps = glob.glob(os.path.join(dirpath, "checkpoint-*/trainer_state.json")) if False else glob.glob(dirpath.replace("/", "\\") + "\\checkpoint-*\\trainer_state.json")
cps = [c for c in cps]
cps.sort(key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
d = json.load(open(cps[-1]))
trace = d.get("log_history", [])
print("total log records:", len(trace))
for h in trace:
    if "loss" in h:
        print(f"step={h['step']:5d} train_loss={h['loss']:.4f}")
    elif "eval_loss" in h:
        print(f"step={h['step']:5d} EVAL loss={h['eval_loss']:.4f} acc={h.get('eval_accuracy')} f1={h.get('eval_macro_f1')}")