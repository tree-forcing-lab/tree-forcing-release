"""Training directories of the k-hop design (family khop4, parent khop3): byte copies of the khop3 stage-1 directories under
the khop4 names, so that jobs/build_bundles.sh (it copies data/student_${v#x1}_*) puts them in the x1khop4 bundle, and
data/eval/khop4/build_report.json. No row is drawn, edited or reordered (the design 1.2).

    python3 scripts/build_khop4.py --khop3-root <E>/src/data --out-root <E>/src/data --report <E>/src/data/eval/khop4

Inputs (all under --khop3-root, the data root of the khop3 build):
- student_khop3_c27/{train,val}.jsonl and student_khop3_k1x/{train,val}.jsonl: md5s must equal
  eval/khop3/build_report.json "md5" and the design values (c27 train 92545dd4..., k1x train d7cc9716..., both val
  9f55fec5...), row counts 27,000 / 393;
- eval/khop/train_kh.jsonl (a16b1520...) and eval/khop/train_k1.jsonl (4bb15f66...), 3,000 rows each, equal to
  eval/khop/build_report.json "md5".
Any other value is a SystemExit before anything is written. --out-root, --report or an output directory that equals an
input directory (resolved paths) is a SystemExit; an output directory that holds a file other than its outputs is a
SystemExit.
Outputs: <out-root>/student_khop4_c27/{train,val}.jsonl, <out-root>/student_khop4_k1x/{train,val}.jsonl (byte copies),
<report>/build_report.json: card, generator, md5 and row counts of the 4 outputs and the 6 inputs, the two replay
equalities (train_kh = rows 0..2,999 of the c27 train file byte for byte; train_k1 = rows 0..2,999 of the k1x train file
byte for byte), the output file sets, inputs unchanged (md5 before = after), d_fail (every failed line) and d_pass.
main exits 2 on a failed line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CARD = "CARD_khop_v4"
SRC = {"c27": "student_khop3_c27", "k1x": "student_khop3_k1x"}
DST = {"c27": "student_khop4_c27", "k1x": "student_khop4_k1x"}
FILES = ("train.jsonl", "val.jsonl")
CARD_MD5 = {"student_khop3_c27/train.jsonl": "92545dd42ca451c39e0e7c104aa78d1e",
            "student_khop3_c27/val.jsonl": "9f55fec541246e41401a7b5fa6c5510a",
            "student_khop3_k1x/train.jsonl": "d7cc9716e5785461d16ec92156104261",
            "student_khop3_k1x/val.jsonl": "9f55fec541246e41401a7b5fa6c5510a",
            "eval/khop/train_kh.jsonl": "a16b1520f0fb8b24ffd8fa57ef696de0",
            "eval/khop/train_k1.jsonl": "4bb15f66c8d8843013eb0410c6a87e48"}
CARD_ROWS = {"student_khop3_c27/train.jsonl": 27000, "student_khop3_c27/val.jsonl": 393,
             "student_khop3_k1x/train.jsonl": 27000, "student_khop3_k1x/val.jsonl": 393,
             "eval/khop/train_kh.jsonl": 3000, "eval/khop/train_k1.jsonl": 3000}
REPLAY = {"train_kh": ("eval/khop/train_kh.jsonl", "student_khop3_c27/train.jsonl"),
          "train_k1": ("eval/khop/train_k1.jsonl", "student_khop3_k1x/train.jsonl")}
N_REPLAY = 3000


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def rows(p: Path) -> int:
    """Lines of a jsonl file (each line ends with a newline; a last line without one is a SystemExit)."""
    b = p.read_bytes()
    if b and not b.endswith(b"\n"):
        raise SystemExit(f"build_khop4: {p} does not end with a newline")
    return b.count(b"\n")


def head_bytes(p: Path, n: int) -> bytes:
    """The first n lines of p, bytes as on disk."""
    out = []
    with open(p, "rb") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            out.append(line)
    return b"".join(out)


def check_inputs(root: Path) -> dict:
    """{rel: md5} of the 6 inputs; SystemExit on a missing file, a wrong md5 or row count, or a md5 that differs from
    the parent reports."""
    rep3_p, repk_p = root / "eval/khop3/build_report.json", root / "eval/khop/build_report.json"
    for p in (rep3_p, repk_p):
        if not p.is_file():
            raise SystemExit(f"build_khop4: parent report {p} is missing")
    try:
        rep3, repk = json.loads(rep3_p.read_text()), json.loads(repk_p.read_text())
    except ValueError as e:
        raise SystemExit(f"build_khop4: a parent report does not parse ({e})")
    if not isinstance(rep3, dict) or not isinstance(repk, dict):
        raise SystemExit("build_khop4: a parent report is not a JSON object")
    m3, mk = rep3.get("md5") or {}, repk.get("md5") or {}
    got = {}
    for rel, want in CARD_MD5.items():
        p = root / rel
        if not p.is_file():
            raise SystemExit(f"build_khop4: input {p} is missing")
        h = md5(p)
        rep_md5 = m3.get(rel) if rel.startswith("student_") else mk.get(rel.split("/")[-1])
        if h != want or rep_md5 != want:
            raise SystemExit(f"build_khop4: {p} md5 {h} (report {rep_md5}), expected {want}")
        n = rows(p)
        if n != CARD_ROWS[rel]:
            raise SystemExit(f"build_khop4: {p} has {n} rows, expected {CARD_ROWS[rel]}")
        got[rel] = h
    return got


def check_paths(root: Path, out_root: Path, report: Path) -> None:
    ins = {(root / d).resolve() for d in SRC.values()} | {(root / "eval/khop3").resolve(), (root / "eval/khop").resolve()}
    outs = {(out_root / d).resolve() for d in DST.values()}
    for p, what in [(out_root, "--out-root"), (report, "--report")] + [(o, "output dir") for o in outs]:
        if p.resolve() in ins:
            raise SystemExit(f"build_khop4: {what} {p} equals an input directory")
    if report.resolve() in outs:
        raise SystemExit(f"build_khop4: --report {report} equals an output directory")
    for o in outs:
        if o.exists() and (not o.is_dir() or set(os.listdir(o)) - set(FILES)):
            raise SystemExit(f"build_khop4: output dir {o} holds other files {sorted(os.listdir(o)) if o.is_dir() else o}")
    if report.exists() and (not report.is_dir() or set(os.listdir(report)) - {"build_report.json"}):
        raise SystemExit(f"build_khop4: report dir {report} holds other files")


def build(root: Path, out_root: Path, report: Path) -> dict:
    check_paths(root, out_root, report)
    before = check_inputs(root)
    fail = []
    replay = {}
    for name, (rp, src) in REPLAY.items():
        a, b = (root / rp).read_bytes(), head_bytes(root / src, N_REPLAY)
        replay[f"{name}_equals_rows_0_2999_of_{src.split('/')[0]}"] = a == b
        if a != b:
            fail.append(f"{name} != rows 0..2,999 of {src}")
    out_md5, out_rows = {}, {}
    for k in SRC:
        od = out_root / DST[k]
        od.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            data = (root / SRC[k] / f).read_bytes()
            (od / f).write_bytes(data)
            rel = f"{DST[k]}/{f}"
            out_md5[rel], out_rows[rel] = md5(od / f), rows(od / f)
            if out_md5[rel] != before[f"{SRC[k]}/{f}"]:
                fail.append(f"{rel} is not a byte copy of {SRC[k]}/{f}")
    # the replay rows are inside the khop4 train files too (D2 of the design 2.5)
    for name, (rp, src) in REPLAY.items():
        dst = out_root / DST["c27" if "c27" in src else "k1x"] / "train.jsonl"
        ok = (root / rp).read_bytes() == head_bytes(dst, N_REPLAY)
        replay[f"{name}_equals_rows_0_2999_of_{dst.parent.name}"] = ok
        if not ok:
            fail.append(f"{name} != rows 0..2,999 of {dst.parent.name}/train.jsonl")
    file_sets = {DST[k]: sorted(os.listdir(out_root / DST[k])) for k in SRC}
    for d, fs in file_sets.items():
        if fs != sorted(FILES):
            fail.append(f"{d} file set {fs}")
    after = {rel: md5(root / rel) for rel in CARD_MD5}
    if after != before:
        fail.append("an input changed during the build")
    rep = {"card": CARD, "generator": "scripts/build_khop4.py",
           "inputs": {rel: {"md5": before[rel], "rows": CARD_ROWS[rel]} for rel in CARD_MD5},
           "md5": out_md5, "rows": out_rows, "file_sets": file_sets, "replay": replay,
           "inputs_unchanged": after == before, "d_fail": fail, "d_pass": not fail}
    report.mkdir(parents=True, exist_ok=True)
    (report / "build_report.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--khop3-root", default="data", help="data root holding student_khop3_* and eval/{khop,khop3}")
    ap.add_argument("--out-root", default="data", help="student_khop4_c27 / student_khop4_k1x are written here")
    ap.add_argument("--report", default="data/eval/khop4")
    a = ap.parse_args(argv)
    rep = build(Path(a.khop3_root), Path(a.out_root), Path(a.report))
    print(json.dumps({k: rep[k] for k in ("md5", "rows", "replay", "inputs_unchanged", "d_pass", "d_fail")}, indent=1))
    if not rep["d_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
