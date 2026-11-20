"""Check the reproduced numbers against those printed in the paper (analysis/expected_numbers.json).
Each entry names the reproduced output file and either a table row (comparison, test set; the first matching row of
the file, whose bold mean is read) or a regex with one group. Tolerance 0.01 points unless the entry gives "tol".
Exit status 1 if any number differs or is missing. python3 analysis/check_numbers.py reproduced/"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "reproduced")
NUM = r"([+\-−]?\d+(?:\.\d+)?)"


def from_row(text, comp, test):
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 5 and cells[0] == comp and cells[1] == test:
            m = re.search(r"\*\*" + NUM + r"\*\*", line)
            return float(m.group(1).replace("−", "-")) if m else None
    return None


fails = 0
for e in json.load(open(ROOT / "analysis/expected_numbers.json")):
    f = OUT / e["file"]
    text = f.read_text() if f.exists() else ""
    if "row" in e:
        got = from_row(text, *e["row"])
    else:
        m = re.search(e["regex"], text)
        got = float(m.group(1).replace("−", "-")) if m else None
    tol = e.get("tol", 0.01)
    ok = got is not None and abs(got - e["paper"]) <= tol + 1e-9
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {e['name']}: paper {e['paper']:+.2f}, reproduced {'missing' if got is None else f'{got:+.2f}'}")
print(f"\n{'all numbers match the paper' if not fails else f'{fails} mismatch(es)'}")
sys.exit(1 if fails else 0)
