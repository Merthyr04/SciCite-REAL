import sys, time, pathlib

run_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("_")
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 900
log = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else run_dir / "train.log"
cache_glob = sys.argv[4] if len(sys.argv) > 4 else None

t0 = time.time()
stage = "WAITING"
while time.time() - t0 < timeout:
    if (run_dir / "metrics.json").exists():
        stage = "METRICS"; break
    if cache_glob and list(pathlib.Path(cache_glob).rglob("augment_*.pkl")):
        stage = "AUG_CACHED"; break
    if log.exists() and "Traceback" in log.read_text(encoding="utf-8", errors="replace"):
        stage = "ERROR"; break
    time.sleep(20)

print(f"STAGE={stage} ELAPSED={time.time()-t0:.0f}s")
if log.exists():
    lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    print("\n".join(lines[-8:]))