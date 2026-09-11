import re
from collections import Counter

for f in [r'e:\Paper\SciCite-REAL\paper\main.tex',
          r'e:\Paper\SciCite-REAL\paper\cvpr\cvpr_main.tex']:
    src = open(f, encoding='utf-8-sig').read()
    begins = re.findall(r'\\begin\{(\w+\*?)\}', src)
    ends = re.findall(r'\\end\{(\w+\*?)\}', src)
    balanced = Counter(begins) == Counter(ends)
    braces = src.count('{') - src.count('}')
    bad_envs = [e for e in set(begins) if begins.count(e) != ends.count(e)]
    name = f.split('\\')[-1]
    print(f"{name}: begin={len(begins)} end={len(ends)} balanced={balanced} "
          f"brace_delta={braces} bad_envs={bad_envs}")
    assert balanced and braces == 0, f"FAIL: {name}"
print("STATIC CHECK PASSED")
