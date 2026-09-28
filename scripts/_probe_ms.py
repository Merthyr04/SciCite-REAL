import urllib.request

def probe(rid, fname="pytorch_model.bin"):
    url = "https://www.modelscope.cn/models/%s/resolve/master/%s" % (rid, fname)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8", "Range": "bytes=0-3"})
        r = urllib.request.urlopen(req, timeout=12)
        cr = r.headers.get("Content-Range", "")
        size = cr.split("/")[-1]
        print("OK  %s / %s  size=%s" % (rid, fname, size))
        return (rid, fname)
    except Exception as e:
        return None

deb = [
    "modelscope/deberta-v3-base",
    "iic/nlp_deberta_v3_large_seq_classify",
    "lixiaolin/deberta-v3-base",
    "stecres/deberta-v3-base",
    "zhihan1996/deberta-v3-base",
    "ljwdeberta/deberta-v3-base",
    "AI-ModelScope/Deberta-v3-base",
    "magec/deberta-v3-base",
    "microsoft/deberta-v3-base",
    "connect/DeBERTa-v3-base",
]
sci = [
    "fs/ai-research/scibert_scivocab_uncased",
    "damo/scibert-basis-uncased",
    "wangxun/scibert_scivocab_uncased",
    "AI-ModelScope/scibert-model",
    "stevengrove/scibert_scivocab_uncased",
    "tongkepro/scibert_scivocab_uncased",
    "allenai/scibert_scivocab_uncased",
]

print("=== DEBERTA ===")
hit = None
for r in deb:
    hit = probe(r)
    if hit:
        break
print("=== SCIBERT ===")
for r in sci:
    hit = probe(r)
    if hit:
        break