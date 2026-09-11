import urllib.request, re

for base in [
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu126/torch/",
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/torch/",
]:
    try:
        html = urllib.request.urlopen(base, timeout=30).read().decode("utf-8", "ignore")
        wins = [l for l in re.findall(r'href="([^"]+)"', html) if "win_amd64" in l and "cp310" in l]
        allwin = [l for l in re.findall(r'href="([^"]+)"', html) if "win_amd64" in l]
        print("=== %s" % base)
        print("cp310 win wheels:")
        for w in wins[-30:]:
            print("   ", w.split("/")[-1], 
                  "SIZE=", re.search(r'data-size="(\d+)"', html) and "?" or "")
        print("recent win names (any py):")
        for w in allwin[-12:]:
            print("    ", w.split("/")[-1])
    except Exception as e:
        print("=== %s ERROR %r" % (base, e)[:160])