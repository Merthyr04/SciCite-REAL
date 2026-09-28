import sys
import time
import urllib.request
from pathlib import Path

MODELS = {
    "deberta": {
        "rid": "microsoft/deberta-v3-base",
        "files": {
            "pytorch_model.bin": "https://www.modelscope.cn/models/microsoft/deberta-v3-base/resolve/master/pytorch_model.bin",
            "config.json": "https://www.modelscope.cn/models/microsoft/deberta-v3-base/resolve/master/config.json",
            "tokenizer.json": "https://www.modelscope.cn/models/microsoft/deberta-v3-base/resolve/master/spm.model",
            "sentencepiece.bpe.model": "https://www.modelscope.cn/models/microsoft/deberta-v3-base/resolve/master/spm.model",
        },
    },
    "scibert": {
        "rid": "allenai/scibert_scivocab_uncased",
        "files": {
            "pytorch_model.bin": "https://www.modelscope.cn/models/allenai/scibert_scivocab_uncased/resolve/master/pytorch_model.bin",
            "config.json": "https://www.modelscope.cn/models/allenai/scibert_scivocab_uncased/resolve/master/config.json",
            "vocab.txt": "https://www.modelscope.cn/models/allenai/scibert_scivocab_uncased/resolve/master/vocab.txt",
        },
    },
    "bert": {
        "rid": "AI-ModelScope/bert-base-uncased",
        "files": {
            "pytorch_model.bin": "https://www.modelscope.cn/models/AI-ModelScope/bert-base-uncased/resolve/master/pytorch_model.bin",
            "config.json": "https://www.modelscope.cn/models/AI-ModelScope/bert-base-uncased/resolve/master/config.json",
            "vocab.txt": "https://www.modelscope.cn/models/AI-ModelScope/bert-base-uncased/resolve/master/vocab.txt",
        },
    },
}


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        total = int(r.headers.get("Content-Length", 0) or 0)
        with open(dest, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    mb = round(total / 1e6, 1)
    print("  ok %s (%.1f MB, %.1f min)" % (dest.name, mb, (time.time() - t0) / 60))
    return total


def main(which, outroot):
    m = MODELS[which]
    out = Path(outroot)
    out.mkdir(parents=True, exist_ok=True)
    for fname, url in m["files"].items():
        try:
            fetch(url, out / fname)
        except Exception as e:
            print("  skip %s failed: %s" % (fname, str(e)[:60]))


if __name__ == "__main__":
    which = sys.argv[1]
    main(which, sys.argv[2])