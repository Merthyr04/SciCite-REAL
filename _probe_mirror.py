import urllib.request, re, sys

for url in [
    "https://mirror.sjtu.edu.cn/pytorch-wheels/",
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu126/",
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/",
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/",
]:
    try:
        html = urllib.request.urlopen(url, timeout=25).read().decode("utf-8", "ignore")
        links = sorted(set(re.findall(r'href=["\']([^"\']+)["\']', html)))
        links = [l for l in links if not l.startswith(("..", "/"))]
        # keep only win_amd64 / index-ish entries
        shown = links[:60]
        print("=== %s (%d entries)" % (url, len(links)))
        for l in shown:
            if l.endswith("win_amd64.whl") or l.endswith("/"):
                print("   ", l)
    except Exception as e:
        print("=== %s ERROR %r" % (url, e)[:160])