"""Unpack the shipped data (data/**/*.jsonl.gz) into the paths the builders, the trainer and the analysis scripts use
(data/**/*.jsonl), and check every file against data/MANIFEST.json (md5 of the uncompressed file).
 python3 scripts/unpack_data.py"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bad = 0
for m in json.load(open(ROOT / "data/MANIFEST.json")):
    out = ROOT / m["file"]
    raw = gzip.open(str(out) + ".gz", "rb").read()
    if hashlib.md5(raw).hexdigest() != m["md5"]:
        print(f"md5 mismatch: {m['file']}"); bad += 1; continue
    out.write_bytes(raw)
print(f"unpacked {len(json.load(open(ROOT / 'data/MANIFEST.json'))) - bad} files" + (f", {bad} mismatches" if bad else ""))
sys.exit(1 if bad else 0)
