import re, pathlib

p = pathlib.Path(r'e:/Paper/SciCite-REAL/paper/scientometrics/sn_scicite.tex')
tex = p.read_text(encoding='utf-8')
lines = tex.split('\n')

for i, ln in enumerate(lines, 1):
    if '\\cite' in ln:
        # find each \cite on this line with local context
        for m in re.finditer(r'\\cite\{[^}]*\}', ln):
            s = max(0, m.start() - 110)
            e = min(len(ln), m.end() + 60)
            print(f'L{i}: ...{ln[s:e]}...')
        print()
