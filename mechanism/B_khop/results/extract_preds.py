#!/usr/bin/env python3
"""Step 1 of 2: read the results tarballs of the k-hop line and write one compact item-level file.

    python3 -I extract_preds.py --results-dir <dir with results_<id>_*.tgz> [--out item_preds.json.gz]

The tarballs are the CHTC job outputs (exp/results/ on the Mac; the same run directories are in the results repo
under runs/). For every job in JOBS the script reads every preds_<set>.jsonl of SETS, takes the arg max of "probs"
as the model's one-pass pick (a letter A-E), and stores the picks as one string per set, in the item order of the
eval file in ../data. It also reads the log (.out) for the code pin, the "stage all: N examples per pass, S steps"
line, the extra-train lines and the weights line, and metrics.json for the seed, the LR and the init model.

Checks (any failure stops the script):
  * every set has exactly the item ids of the eval file and the same gold label per item;
  * no NaN / missing probs, 5 options per row;
  * the seed in metrics.json equals the seed in JOBS (eval-only jobs: the seed of the evaluated weights).
Standard library only.
"""
import argparse, glob, gzip, io, json, math, os, re, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

KHOP = ["khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_wp_rev", "khop_vb_fwd", "khop_vb_rev", "khop_cag_fwd",
        "khop_ca_fwd_qs", "khop_blind"]
HINT = ["hint_plain"] + [f"hint_true_r{r}" for r in range(1, 7)]
PLAIN = ["gsm8k", "gsm8kv2", "gsm1k", "gsm8kx", "gsm1kx", "valx"]
SETS = KHOP + HINT + PLAIN

# job id -> (arm, seed). Eval-only jobs (750-755, 771) evaluate saved weights; their seed is the training seed of
# those weights (751 w663 s16, 752 w664 s17, 753 w665 s18, 754 w540 s19, 755 w666 s20, 771 w667 s16; 750 base: 0).
JOBS = {750: ("BASE0", 0), 751: ("A2Z", 16), 752: ("A2Z", 17), 753: ("A2Z", 18), 754: ("A2Z", 19), 755: ("A2Z", 20),
        771: ("A2Z96", 16)}
for i, s in enumerate(range(16, 21)):
    JOBS[756 if s == 16 else 758 + s - 17] = ("KH", s)
    JOBS[757 if s == 16 else 762 + s - 17] = ("K1", s)
    JOBS[766 + i] = ("KD", s)
    JOBS[875 + i] = ("D27", s)
    JOBS[880 + i] = ("R3", s)
    JOBS[885 + i] = ("H3", s)
    JOBS[975 if s == 16 else 1029 + s - 17] = ("CUR1", s)
    JOBS[985 if s == 16 else 1033 + s - 17] = ("K1X1", s)
    JOBS[1027 if s == 16 else 1037 + s - 17] = ("CUR-R", s)
    JOBS[1028 if s == 16 else 1041 + s - 17] = ("K1X-R", s)
JOBS.update({980: ("CUR-A2", 16), 990: ("K1X-A2", 16), 1025: ("CUR-L", 16), 1026: ("K1X-L", 16)})
# A2-6k references (answer-only GSM, no chain rows). s19 is j540 in both sets (run_khop.sh), so A2re s19 = A2orig s19.
for s, j in zip(range(16, 21), (520, 525, 530, 540, 542)):
    JOBS.setdefault(j, ("A2orig", s))
REF_RE = {16: 663, 17: 664, 18: 665, 19: 540, 20: 666}
JOBS.update({663: ("A2re", 16), 664: ("A2re", 17), 665: ("A2re", 18), 666: ("A2re", 20), 667: ("A2-96k", 16)})

EVAL_FILES = {s: os.path.join(DATA, "eval", "khop", s + ".jsonl") for s in KHOP}
EVAL_FILES.update({s: os.path.join(DATA, "eval", "hint", s + ".jsonl") for s in HINT})


def tarball(results_dir, j):
    fs = [f for f in glob.glob(os.path.join(results_dir, f"results_{j}_*.tgz")) if "/results_smoke_" not in f]
    if len(fs) != 1:
        sys.exit(f"extract: need exactly one tarball for j{j}, found {fs}")
    return fs[0]


def read_eval(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r["item_id"] for r in rows], "".join(r["gold_label"] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "item_preds.json.gz"))
    a = ap.parse_args()
    ref = {s: read_eval(p) for s, p in EVAL_FILES.items()}   # khop + hint: order and gold from ../data
    out = {"sets": {}, "jobs": {}}
    for j in sorted(JOBS):
        arm, seed = JOBS[j]
        tf = tarball(a.results_dir, j)
        preds, metrics, log = {}, None, ""
        with tarfile.open(tf) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                mm = re.search(r"/preds_([a-z0-9_]+)\.jsonl$", m.name)
                if mm and mm.group(1) in SETS:
                    rows = [json.loads(l) for l in io.BytesIO(t.extractfile(m).read()) if l.strip()]
                    preds[mm.group(1)] = rows
                elif m.name.endswith("/metrics.json"):
                    metrics = json.load(t.extractfile(m))
                elif re.search(r"^logs/[^/]+\.out$", m.name):
                    log += t.extractfile(m).read().decode("utf-8", "replace")
        rec = {"arm": arm, "seed": seed, "tarball": os.path.basename(tf), "run_dir": None, "picks": {}}
        for s, rows in preds.items():
            for r in rows:
                p = r["probs"]
                if len(p) != 5 or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in p):
                    sys.exit(f"extract: j{j} {s} {r['item_id']}: bad probs {p}")
            ids, gold = [r["item_id"] for r in rows], "".join(r["gold_label"] for r in rows)
            if s in ref:
                rid, rgold = ref[s]
                pos = {i: n for n, i in enumerate(ids)}
                if sorted(ids) != sorted(rid) or len(pos) != len(ids):
                    sys.exit(f"extract: j{j} {s}: item ids differ from {EVAL_FILES[s]}")
                rows = [rows[pos[i]] for i in rid]
                if "".join(r["gold_label"] for r in rows) != rgold:
                    sys.exit(f"extract: j{j} {s}: gold labels differ from the eval file")
                ids, gold = rid, rgold
            else:   # plain GSM sets: the first job fixes the order; every later job must match it
                if s not in out["sets"]:
                    out["sets"][s] = {"ids": ids, "gold": gold}
                pos = {i: n for n, i in enumerate(ids)}
                ids0 = out["sets"][s]["ids"]
                if sorted(ids) != sorted(ids0):
                    sys.exit(f"extract: j{j} {s}: item ids differ from the first job")
                rows = [rows[pos[i]] for i in ids0]
                if "".join(r["gold_label"] for r in rows) != out["sets"][s]["gold"]:
                    sys.exit(f"extract: j{j} {s}: gold labels differ from the first job")
            out["sets"].setdefault(s, {"ids": ids, "gold": gold})
            rec["picks"][s] = "".join(r["labels"][max(range(5), key=r["probs"].__getitem__)] for r in rows)
        args = (metrics or {}).get("args") or {}
        rec["run_dir"] = os.path.basename(os.path.normpath(args.get("out", ""))) or None
        rec["init_model"] = args.get("model")
        rec["lr"] = args.get("lr")
        rec["epochs"] = args.get("epochs")
        rec["train"] = args.get("train")
        rec["extra_train"] = args.get("extra_train") or None
        rec["metrics_seed"] = args.get("seed")
        if args.get("seed") is not None and arm not in ("A2Z", "A2Z96", "BASE0") and int(args["seed"]) != seed:
            sys.exit(f"extract: j{j}: metrics seed {args['seed']} != {seed}")
        hist = (metrics or {}).get("history") or []
        rec["last_step"] = hist[-1]["step"] if hist and "step" in hist[-1] else None
        pin = re.findall(r"^code pin: (\S+)", log, re.M)
        rec["code_pin"] = pin[0] if pin else None
        st = re.findall(r"^stage (\S+): (\d+) examples per pass, (\d+) steps", log, re.M)
        rec["stages"] = [{"stage": x, "examples": int(n), "steps": int(k)} for x, n, k in st]
        rec["extra_train_log"] = re.findall(r"^extra-train: (\d+) rows from (\S+)", log, re.M)
        w = re.findall(r"^== weights -> (\S+)", log, re.M)
        rec["weights_saved_to"] = w[0] if w else None
        out["jobs"][str(j)] = rec
        print(f"j{j} {arm:7s} s{seed} sets {len(rec['picks'])} steps {[x['steps'] for x in rec['stages']]} pin {rec['code_pin']}")
    with gzip.GzipFile(a.out, "wb", mtime=0) as f:
        f.write(json.dumps(out, sort_keys=True).encode())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
