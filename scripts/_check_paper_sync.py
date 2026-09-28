import re

files = {
    'main.tex': r'e:\Paper\SciCite-REAL\paper\main.tex',
    'cvpr_main.tex': r'e:\Paper\SciCite-REAL\paper\cvpr\cvpr_main.tex',
    'paper.html': r'e:\Paper\SciCite-REAL\paper\paper.html',
}
stale = ['120', '160.0', '151.4', r'3.2\times', '3e-14', '1e-14',
         'six augmented', '~5 s', '<5 s', 'under 5 sec', 'under 4 sec']
for name, path in files.items():
    text = open(path, encoding='utf-8').read()
    problems = [s for s in stale if s in text]
    print(f'{name}: stale = {problems if problems else "none"}')
    dollars = len(re.findall(r'(?<!\\)\$', text))
    bal = 'balanced' if dollars % 2 == 0 else 'UNBALANCED'
    print(f'  $ count: {dollars} ({bal})')
    braces = 0
    for ch in text:
        if ch == '{':
            braces += 1
        elif ch == '}':
            braces -= 1
    print(f'  brace balance: {braces} ({"ok" if braces == 0 else "UNBALANCED"})')
    # new numbers present?
    for needle in ['76', '11.6', '34.1', '153.5', '2.1M', '48 tokens']:
        if needle not in text:
            print(f'  MISSING new number: {needle}')
