from pypdf import PdfReader
import re

r = PdfReader(r'e:/Paper/SciCite-REAL/paper/scientometrics/sn_scicite.pdf')
t = '\n'.join(p.extract_text() or '' for p in r.pages)
t = re.sub(r'-\n', '', t)
t = re.sub(r'\s+', ' ', t)

print('=== doubled "X et al. X et al. (" ===')
d1 = re.findall(r'([A-Z][a-zA-Z]+ et al\.) \1 \(', t)
print(d1 or 'NONE')
print('=== doubled "X and Y X and Y (" ===')
d2 = re.findall(r'([A-Z][a-zA-Z]+ and [A-Z][a-zA-Z]+) \1 \(', t)
print(d2 or 'NONE')

print('=== sample \citet renderings ===')
for kw in ['Teufel', 'Valenzuela', 'Jurgens', 'Cohan', 'Garfield', 'Small', 'Cronin',
           'Bornmann', 'Tahamtan', 'Zhu', 'Mingers', 'Pride', 'Kunnath']:
    for m in re.finditer(r'\b' + kw, t):
        seg = t[max(0, m.start()-40):m.start()+90]
        print(f'  {kw}: ...{seg}...')
        break

print('=== sample \citep renderings ===')
for kw in ['SciCite (', 'ACL-ARC (', 'ELMo', 'REALM', 'BM25 ']:
    for m in re.finditer(re.escape(kw), t):
        print(f'  {kw}: ...{t[max(0,m.start()-30):m.start()+110]}...')
        break
