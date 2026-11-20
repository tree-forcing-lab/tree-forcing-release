"""scripts/build_khop3.py and the two-stage A2 path of the curriculum.
Real-file tests (the parent files: KHOP_EVAL_DIR, KHOP2_EVAL_DIR, GSMVT_DIR; defaults next to E/src/data):
- refusals: a changed train_kd27, train_k1, val.jsonl or khop parent file, and --out-root / --report equal to an input
  directory, each stop the build (SystemExit) before anything is written;
- the replay (build_khop.build_train seed 1 + build_khop2.new_rows seed 3, Item.row wrapped) equals train_kd27 (27,000
  rows) and train_k1 (3,000 rows); a replay fixture with one row changed (in either stream) stops the build;
- c27 files equal the inputs byte for byte; K1X rows 0..2,999 equal train_k1 byte for byte; for every K1X row the
  statements equal the kd27 twin's, the query is the target chain's hop-1 variable and the gold its value (parsed by an
  independent walk here), the option rule holds (ca / wp), vb options are the five roots; gold shares within the bars;
  eval prompt-hash overlap 0; item_ids distinct; the report values recomputed from the files equal the report;
  deterministic (two builds byte-equal, and equal to the files in the data dir when they exist); the parent files are
  unchanged (md5 before and after, against the design values).
- the two-stage tiny run (the tests/test_extra_train.py 1-rank FSDP pattern: torchrun --local-addr 127.0.0.1, cpu-adamw,
  --eval-value-render vt, label smoothing 0.1, --token-budget 4096): stage 1 on 18 khop rows with --save, the with
  --model set to the saved weights on 6 GSM rows, one extra eval set and --eval-at 0.5. A test
  wrapper (written to a temp dir; the trainer is not changed) records the examples of every training kl_loss call
  (item_id, text with the option order, labels, target, weight). the equals the reference A2 runs from the same
  stage-1 weights without the extra set (with and without --eval-at): same examples per micro-batch, same per-update
  losses (exact), same final weights; the stage-2 examples also equal an A2 run from the base tiny model; final steps
  ceil(18 / 4) = 5 and ceil(6 / 4) = 2; losses finite.
Run: python -m unittest tests.test_build_khop3 (from the repo)."""
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import build_khop as B  # noqa: E402
import build_khop2 as B2  # noqa: E402
import build_khop3 as K3  # noqa: E402

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from test_depth_forcing import gpt2_tok, tiny_qwen35
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

E_DEFAULT = ROOT.parents[1]                     # E (repo = E/src_khop3/repo)
KHOP = Path(os.environ.get("KHOP_EVAL_DIR") or E_DEFAULT / "src/data/eval/khop")
KHOP2 = Path(os.environ.get("KHOP2_EVAL_DIR") or E_DEFAULT / "src/data/eval/khop2")
GSMVT = Path(os.environ.get("GSMVT_DIR") or KHOP2.parents[1] / "student_gsm8k_tf_vt")
DATA_ROOT = KHOP2.parents[1]
HAVE_DATA = all((p / f).exists() for p, f in ((KHOP2, "train_kd27.jsonl"), (KHOP, "train_k1.jsonl"),
                                              (KHOP, "build_report.json"), (GSMVT, "val.jsonl")))
# the design 1.1 / 1.2 md5s
CARD_MD5 = {"train_kd27": "92545dd42ca451c39e0e7c104aa78d1e", "train_k1": "4bb15f66c8d8843013eb0410c6a87e48",
            "val": "9f55fec541246e41401a7b5fa6c5510a"}
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
TRAIN = str(ROOT / "scripts/train_student.py")
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
            "--local-addr", "127.0.0.1"]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def inputs_md5():
    fs = [KHOP2 / "train_kd27.jsonl", KHOP2 / "build_report.json", GSMVT / "val.jsonl", KHOP / "build_report.json"] + \
         [KHOP / f"{s}.jsonl" for s in B2.PARENT_FILES]
    return {str(p): md5(p) for p in fs}


def copy_inputs(d: Path):
    """Copies of the three input dirs (only the files the build reads) under d: (khop2, khop, gsm)."""
    k2, k1, g = d / "in/khop2", d / "in/khop", d / "in/gsm"
    for x in (k2, k1, g):
        x.mkdir(parents=True)
    for f in ("train_kd27.jsonl", "build_report.json"):
        shutil.copy(KHOP2 / f, k2 / f)
    for s in list(B2.PARENT_FILES) + ["build_report"]:
        shutil.copy(KHOP / f"{s}.jsonl" if s != "build_report" else KHOP / "build_report.json", k1)
    shutil.copy(GSMVT / "val.jsonl", g / "val.jsonl")
    return k2, k1, g


def walk_hop1(prompt):
    """(query name, the target chain's hop-1 variable name, values) by a walk independent of build_khop.parse_prompt's
    chain_of: statements split on '. ', a root is a statement whose right side has no variable."""
    head = prompt.partition("\nOptions:\n")[0][len("Question: "):]
    m = re.search(r"(What is the value of ([a-z]{2})\?|How many \w+ does (\w+) have\?)$", head)
    q = m.group(2) or m.group(3)
    parent = {}
    for s in re.split(r"(?<=\.) ", head[:m.start()].strip()):
        ca = re.match(r"^([a-z]{2}) = ([a-z]{2})?", s)
        wp = re.match(r"^(\w+) has .* (?:than|as) (\w+)\.$", s)
        if ca:
            parent[ca.group(1)] = ca.group(2)
        elif wp:
            parent[wp.group(1)] = wp.group(2)
        else:
            parent[re.match(r"^(\w+) has", s).group(1)] = None
    chain = [q]
    while parent[chain[-1]] is not None:
        chain.append(parent[chain[-1]])
    return q, (chain[-2] if len(chain) >= 2 else None), chain


@unittest.skipUnless(HAVE_DATA, "needs the khop / khop2 / gsmvt files")
class TestRealBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md5_before = inputs_md5()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.rep = K3.build(KHOP2, KHOP, GSMVT, cls.tmp / "a/data", cls.tmp / "a/report", tok=None)
        cls.rep_b = K3.build(KHOP2, KHOP, GSMVT, cls.tmp / "b/data", cls.tmp / "b/report", tok=None)
        cls.kd27 = [json.loads(l) for l in open(KHOP2 / "train_kd27.jsonl")]
        cls.k1 = [json.loads(l) for l in open(KHOP / "train_k1.jsonl")]
        cls.out = cls.tmp / "a/data"
        cls.k1x_lines = (cls.out / "student_khop3_k1x/train.jsonl").read_bytes().splitlines(keepends=True)
        cls.k1x = [json.loads(l) for l in cls.k1x_lines]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_parent_files_unchanged(self):
        self.assertEqual(inputs_md5(), self.md5_before)
        self.assertEqual(md5(KHOP2 / "train_kd27.jsonl"), CARD_MD5["train_kd27"])
        self.assertEqual(md5(KHOP / "train_k1.jsonl"), CARD_MD5["train_k1"])
        self.assertEqual(md5(GSMVT / "val.jsonl"), CARD_MD5["val"])
        prep = json.loads((KHOP / "build_report.json").read_text())
        for s in B2.PARENT_FILES:
            self.assertEqual(md5(KHOP / f"{s}.jsonl"), prep["md5"][f"{s}.jsonl"], s)

    def test_report_passes(self):
        self.assertTrue(self.rep["d_pass"], self.rep["d_fail"])
        self.assertEqual(self.rep["replay"], {"kd27_rows_equal": 27000, "kd27_rows": 27000, "kd27_replayed": 27000,
                                              "k1_rows_equal": 3000, "k1_rows": 3000, "k1_replayed": 3000,
                                              "ids_rendered_more_than_once": 0})
        self.assertEqual(self.rep["stream_params"], {"train_seed": 1, "n_kh": 1000, "n_kd_extra": 2000,
                                                     "khop2_seed": 3, "new_per_task": 6000})
        self.assertEqual(self.rep["updates"], {"train_rows": 27000, "updates_mbs2_ga16": 844})

    def test_c27_and_val_byte_equal(self):
        self.assertEqual((self.out / "student_khop3_c27/train.jsonl").read_bytes(), (KHOP2 / "train_kd27.jsonl").read_bytes())
        for d in ("student_khop3_c27", "student_khop3_k1x"):
            self.assertEqual((self.out / d / "val.jsonl").read_bytes(), (GSMVT / "val.jsonl").read_bytes())
            self.assertEqual(sorted(x.name for x in (self.out / d).iterdir()), ["train.jsonl", "val.jsonl"])

    def test_k1x_rows(self):
        self.assertEqual(len(self.k1x), 27000)
        self.assertEqual(self.k1x_lines[:3000], (KHOP / "train_k1.jsonl").read_bytes().splitlines(keepends=True))
        ev_hash = {B.phash(json.loads(l)["prompt"]) for s in B.EVAL_FILES for l in open(KHOP / f"{s}.jsonl")}
        shares = defaultdict(Counter)
        for a, t in zip(self.kd27, self.k1x):
            self.assertEqual(t["item_id"], a["item_id"] + "-k1")
            self.assertEqual((t["group"], t["depth"]), (a["group"], a["depth"]))
            ha, ht = a["prompt"].partition("\nOptions:\n")[0], t["prompt"].partition("\nOptions:\n")[0]
            self.assertEqual(ha[:ha.rindex(". ") + 1], ht[:ht.rindex(". ") + 1], t["item_id"])   # same statements
            qa, h1a, _ = walk_hop1(a["prompt"])
            qt, _, _ = walk_hop1(t["prompt"])
            self.assertEqual(qt, h1a, t["item_id"])                                             # hop-1 query
            P = B.parse_prompt(t["prompt"])
            gold = P["options"]["ABCDE".index(t["gold_label"])]
            self.assertEqual(gold, P["values"][qt], t["item_id"])
            task = B.task_of(t)
            if task in ("ca", "wp"):
                self.assertTrue(B.option_set_ok(P["options"], set(P["literals"])), t["item_id"])
            else:
                self.assertEqual(sorted(P["options"]), sorted(s["c"] for s in P["stmts"] if s["op"] == "root"))
            self.assertEqual(t["teacher"], {L: float(L == t["gold_label"]) for L in "ABCDE"})
            self.assertNotIn(B.phash(t["prompt"]), ev_hash)
            shares[(task, t["depth"])][t["gold_label"]] += 1
        for c, cn in shares.items():
            for L in "ABCDE":
                self.assertTrue(0.18 <= cn[L] / sum(cn.values()) <= 0.22, (c, L, cn))
        self.assertEqual(len({r["item_id"] for r in self.k1x}), 27000)

    def test_report_recomputed(self):
        r, out = self.rep, self.out
        for f, v in r["md5"].items():
            self.assertEqual(md5(out / f), v, f)
        for f, v in r["rows"].items():
            self.assertEqual(len((out / f).read_bytes().splitlines()), v, f)
        self.assertEqual(r["md5"]["student_khop3_c27/train.jsonl"], CARD_MD5["train_kd27"])
        self.assertEqual(r["inputs"]["train_k1.jsonl"]["md5"], CARD_MD5["train_k1"])
        cn = Counter(t["gold_label"] for t in self.k1x)
        self.assertEqual(r["k1x_checks"]["gold_counts"]["all"], {L: cn[L] for L in "ABCDE"})
        kd_hash = {B.phash(x["prompt"]) for x in self.kd27}
        shared = [t for t in self.k1x if B.phash(t["prompt"]) in kd_hash]
        self.assertEqual(r["kd27_hash_overlap"]["k1x_rows"], len(shared))
        self.assertTrue(all(t["depth"] == 1 for t in shared))
        self.assertEqual(r["token_totals"]["kd27"], sum(len(x["prompt"]) for x in self.kd27))   # tok=None: characters
        self.assertEqual(r["token_totals"]["k1x"], sum(len(x["prompt"]) for x in self.k1x))
        self.assertEqual(r["eval_hash_overlap"], {"k1x": 0, "c27": 0})
        for x in ("statement_mismatch", "query_not_hop1", "gold_not_hop1_value", "option_rule_fail",
                  "vb_options_not_roots", "row_rule_errors", "teacher_not_onehot", "keyset_differs"):
            self.assertEqual(r["k1x_checks"][x], 0, x)

    def test_deterministic_and_equal_to_data_dir(self):
        for d in ("student_khop3_c27", "student_khop3_k1x"):
            for f in ("train.jsonl", "val.jsonl"):
                a, b = self.tmp / "a/data" / d / f, self.tmp / "b/data" / d / f
                self.assertEqual(md5(a), md5(b), f"{d}/{f}")
                real = DATA_ROOT / d / f
                if real.exists():                         # the bundle's files are this build's files
                    self.assertEqual(md5(real), md5(a), str(real))
        real_rep = DATA_ROOT / "eval/khop3/build_report.json"
        if real_rep.exists():
            rr = json.loads(real_rep.read_text())
            self.assertEqual(rr["md5"], self.rep["md5"])
            self.assertTrue(rr["d_pass"], rr["d_fail"])
            self.assertEqual(rr["token_totals"]["unit"], "tokens")     # the data-dir build used the recorded tokenizer
            self.assertLessEqual(abs(rr["token_totals"]["rel_k1x_minus_kd27"]), 0.01)
            self.assertEqual(sorted(p.name for p in real_rep.parent.iterdir()), ["build_report.json"])


@unittest.skipUnless(HAVE_DATA, "needs the khop / khop2 / gsmvt files")
class TestReplayAndRefusals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        reps = K3.check_inputs(KHOP2, KHOP, GSMVT)
        cls.rk, cls.r1, cls.rec, cls.calls = K3.replay(KHOP, K3.stream_params(reps))
        cls.kd27 = [json.loads(l) for l in open(KHOP2 / "train_kd27.jsonl")]
        cls.k1 = [json.loads(l) for l in open(KHOP / "train_k1.jsonl")]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_replay_equal_and_changed_row_stops(self):
        res = K3.check_replay(self.rk, self.r1, self.kd27, self.k1)
        self.assertEqual(res["kd27_rows_equal"], 27000)
        self.assertEqual(res["k1_rows_equal"], 3000)
        self.assertEqual(len(self.rec), 27000 + 3000)           # every kd27 id and every -k1 id recorded
        for which, i in (("kd", 5), ("kd", 20000), ("k1", 7)):
            rk, r1 = list(self.rk), list(self.r1)
            if which == "kd":
                rk[i] = dict(rk[i], gold_label="E" if rk[i]["gold_label"] != "E" else "A")
            else:
                r1[i] = dict(r1[i], prompt=r1[i]["prompt"] + " ")
            with self.assertRaises(SystemExit) as c:
                K3.check_replay(rk, r1, self.kd27, self.k1)
            self.assertIn("the replay does not reproduce", str(c.exception))
        with self.assertRaises(SystemExit):                     # a shorter replay stops it too
            K3.check_replay(self.rk[:-1], self.r1, self.kd27, self.k1)

    def test_last_call_wins(self):
        """A redrawn khop2 row (row() called again for the same id after a prompt-hash clash) maps to the accepted Item:
        the recorded Item renders the kept row's statements."""
        # on the real streams no row was redrawn after row (every redraw was a value-key redraw), so the rule is moot
        # here; the check below would catch a first-call-wins mapping if one did happen
        self.assertEqual(sum(v > 1 for v in self.calls.values()), 0)
        bad = []
        for r in self.kd27:                                     # every row, both streams
            it = self.rec[r["item_id"]]
            order = r["group"].split("/")[2]
            if "Question: " + " ".join(it.statements(order)) + " " != K3.stmt_text(r["prompt"]):
                bad.append(r["item_id"])
        self.assertEqual(bad, [])

    def test_refusals(self):
        cases = [("kd27", "md5 of"), ("k1", "md5 of"), ("val", "has md5"), ("parent", "md5 of"),
                 ("out_root", "must not be an input"), ("report", "must not be an input")]
        for what, msg in cases:
            d = self.tmp / what
            k2, k1, g = copy_inputs(d)
            out_root, rep = d / "out", d / "rep"
            if what == "kd27":
                with open(k2 / "train_kd27.jsonl", "a") as f:
                    f.write("\n")
            elif what == "k1":
                with open(k1 / "train_k1.jsonl", "a") as f:
                    f.write("\n")
            elif what == "val":
                with open(g / "val.jsonl", "a") as f:
                    f.write("\n")
            elif what == "parent":
                with open(k1 / "khop_ca_fwd.jsonl", "a") as f:
                    f.write("\n")
            elif what == "out_root":
                out_root = k2
            else:
                rep = g
            with self.assertRaises(SystemExit) as c:
                K3.build(k2, k1, g, out_root, rep, tok=None)
            self.assertIn(msg, str(c.exception), what)
            self.assertFalse((out_root / "student_khop3_k1x").exists() if what != "out_root" else False, what)
            self.assertFalse((rep / "build_report.json").exists(), what)

    def test_extra_file_in_output_dir_refused(self):
        d = self.tmp / "extra"
        (d / "out/student_khop3_c27").mkdir(parents=True)
        (d / "out/student_khop3_c27/stale.jsonl").write_text("{}\n")
        with self.assertRaises(SystemExit) as c:
            K3.build(KHOP2, KHOP, GSMVT, d / "out", d / "rep", tok=None)
        self.assertIn("holds other files", str(c.exception))


# --- the two-stage A2 path on the tiny model ---------------------------------------------------------------------------
TRACER = r'''
import importlib.util, json, os, sys
import torch
spec = importlib.util.spec_from_file_location("train_student", sys.argv[1])
T = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[1]] + sys.argv[2:]
spec.loader.exec_module(T)
rec = []
orig = T.kl_loss
def kl_loss(z, mb, *a, **k):
    if torch.is_grad_enabled():           # training calls only (evaluation runs under no_grad)
        rec.append([[e.item_id, e.text, list(e.labels), [float(x) for x in e.target], float(e.weight)] for e in mb])
    return orig(z, mb, *a, **k)
T.kl_loss = kl_loss
try:
    T.main()
finally:
    if os.environ.get("RANK", "0") == "0":
        open(os.environ["KHOP3_TRACE"], "w").write(json.dumps(rec))
'''


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestTwoStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tok = gpt2_tok()
        except Exception as e:
            raise unittest.SkipTest(f"no local gpt2 tokenizer: {e}")
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        tiny_qwen35(len(cls.tok)).save_pretrained(d / "m")
        cls.tok.save_pretrained(d / "m")
        gsm = json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())["gsm8k"]
        kh, _, _, _, _ = B.build_train(random.Random(1), defaultdict(Counter), 6, 0)        # 18 khop rows
        assert len(gsm) == 6 and len(kh) == 18
        w = lambda name, rows: (d / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        w("gsm.jsonl", gsm)
        w("kh.jsonl", kh)
        w("ev.jsonl", gsm[:3] + kh[:5])
        w("kx.jsonl", kh[5:9])
        (d / "tracer.py").write_text(TRACER)
        cls.s1 = cls.train_run(d / "s1", d / "m", d / "kh.jsonl", save=True)
        cls.runs = {}
        s1m = d / "s1/model"
        cls.runs["S2"] = cls.train_run(d / "s2", s1m, d / "gsm.jsonl", extra_eval=True, eval_at=True, save=True)
        cls.runs["A2w"] = cls.train_run(d / "a2w", s1m, d / "gsm.jsonl", eval_at=True, save=True)
        cls.runs["A2w_noat"] = cls.train_run(d / "a2w_noat", s1m, d / "gsm.jsonl", save=True)
        cls.runs["A2base"] = cls.train_run(d / "a2base", d / "m", d / "gsm.jsonl", eval_at=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def train_run(cls, out, model, train, extra_eval=False, eval_at=False, save=False):
        ev = ["val=" + str(cls.d / "ev.jsonl")] + (["kx=" + str(cls.d / "kx.jsonl")] if extra_eval else [])
        trace = str(out) + ".trace.json"
        cmd = [*TORCHRUN, str(cls.d / "tracer.py"), TRAIN, "--model", str(model), "--train", str(train), "--eval", *ev,
               "--eval-subq", str(cls.d / "gsm.jsonl"), "--eval-value-render", "vt", "--final", "teacher", "--subq",
               "none", "--epochs", "1", "--micro-bs", "2", "--grad-accum", "2", "--max-len", "512", "--eval-max-len",
               "512", "--precision", "fp32master", "--lr", "1e-3", "--log-every", "1", "--seed", "16", "--order-seed",
               "16", "--fsdp", "--fsdp-optim", "cpu-adamw", "--label-smoothing", "0.1", "--token-budget", "4096",
               "--out", str(out)]
        if eval_at:
            cmd += ["--eval-at", "0.5", "--eval-mid", "val"]
        if save:
            cmd += ["--save"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env={**ENV, "KHOP3_TRACE": trace})
        if r.returncode != 0:
            raise AssertionError(f"{out.name}: exit {r.returncode}\n{r.stderr[-3000:]}")
        m = json.loads((out / "metrics.json").read_text())
        return {"stdout": r.stdout, "metrics": m, "trace": json.loads(Path(trace).read_text()), "out": out}

    def weights(self, out):
        from safetensors.torch import load_file
        fs = sorted((out / "model").glob("*.safetensors"))
        self.assertTrue(fs, out)
        sd = {}
        for f in fs:
            sd.update(load_file(str(f)))
        return sd

    def test_stage1(self):
        m = self.s1["metrics"]
        self.assertIn("stage all: 18 examples per pass, 5 steps", self.s1["stdout"])
        self.assertEqual(m["history"][-1]["step"], math.ceil(18 / 4))
        self.assertTrue(all(math.isfinite(x["loss"]) for x in m["history"]))
        self.assertTrue((self.d / "s1/model/config.json").exists())
        # 5 updates x 4 sequences: the 18 rows once, the last update filled from the next shuffled pass (the trainer's
        # rule for every A2 job: 844 x 32 = 27,008 at S1, 186 x 32 = 5,952 at S2)
        ids = [e[0] for mb in self.s1["trace"] for e in mb]
        self.assertEqual(len(ids), 5 * 4)
        self.assertEqual(set(ids), {json.loads(l)["item_id"] for l in (self.d / "kh.jsonl").read_text().splitlines()})
        self.assertEqual(len(set(ids[:16])), 16)                   # updates 1-4: first-pass rows only
        self.assertEqual((m["args"]["seed"], m["args"]["order_seed"]), (16, 16))

    def test_stage2_equals_a2_from_the_same_weights(self):
        s2 = self.runs["S2"]
        self.assertIn("stage all: 6 examples per pass, 2 steps", s2["stdout"])
        self.assertEqual(s2["metrics"]["history"][-1]["step"], math.ceil(6 / 4))
        self.assertIn("kx", s2["metrics"]["eval"])                      # the extra set was evaluated
        self.assertEqual([x["step"] for x in s2["metrics"]["trajectory"]], [1])   # the --eval-at 0.5 point ran
        sw = self.weights(s2["out"])
        for ref in ("A2w", "A2w_noat"):
            r = self.runs[ref]
            self.assertNotIn("kx", r["metrics"]["eval"])
            self.assertEqual(len(r["metrics"].get("trajectory") or []), 1 if ref == "A2w" else 0, ref)
            self.assertEqual(s2["trace"], r["trace"], ref)               # item_ids, texts (option order), targets
            self.assertEqual([x["loss"] for x in s2["metrics"]["history"]],
                             [x["loss"] for x in r["metrics"]["history"]], ref)
            self.assertTrue(all(math.isfinite(x["loss"]) for x in r["metrics"]["history"]))
            rw = self.weights(r["out"])
            self.assertEqual(sorted(sw), sorted(rw))
            for k in sw:
                self.assertTrue(torch.equal(sw[k], rw[k]), (ref, k))
        self.assertGreater(len(s2["trace"]), 0)

    def test_stage2_batches_equal_a2_from_base(self):
        s2, base = self.runs["S2"], self.runs["A2base"]
        self.assertEqual(s2["trace"], base["trace"])                    # only the starting weights differ
        self.assertNotEqual([x["loss"] for x in s2["metrics"]["history"]],
                            [x["loss"] for x in base["metrics"]["history"]])   # the trace is not trivially equal


if __name__ == "__main__":
    unittest.main()
