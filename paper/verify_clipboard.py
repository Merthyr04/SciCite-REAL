import subprocess

ps = "(Get-Clipboard -Raw) -replace \"`r`n\", \"`n\""
r = subprocess.run(["powershell", "-Command", ps], capture_output=True, text=True, encoding="utf-8")
clip = r.stdout.replace("\r\n", "\n")
orig = open(r"e:\Paper\SciCite-REAL\paper\main.tex", encoding="utf-8-sig").read().replace("\r\n", "\n")
clip_n = clip.rstrip("\n")
orig_n = orig.rstrip("\n")
print("orig len:", len(orig_n))
print("clip len:", len(clip_n))
print("match:", clip_n == orig_n)
if clip_n != orig_n:
    for i, (a, b) in enumerate(zip(orig_n, clip_n)):
        if a != b:
            print("first diff at", i)
            print("orig:", repr(orig_n[max(0, i-30):i+30]))
            print("clip:", repr(clip_n[max(0, i-30):i+30]))
            break
    else:
        print("length mismatch only")
