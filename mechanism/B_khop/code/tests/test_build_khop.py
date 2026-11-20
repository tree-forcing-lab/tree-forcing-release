"""scripts/build_khop.py. Standard library only.
Generator level (a small build in a temp dir): determinism by seed; the option rule and the shared
skeleton (one root value and one op-kind sequence for the five chains, hop 1 + / - with five distinct constants); the
parser on hand-written rows; the baselines' uniform-pick convention, the chase lines and the hard flag on a hand row;
row_rule_ok refuses a row whose chains do not share the skeleton.
The real files (E/src/data/eval/khop, or KHOP_EVAL_DIR; skipped when absent): row counts per cell (1.2); the row key set
equals the 32 gsmvt keys (training rows) plus task, order, k, label_order, gold_probs (eval rows); gold in options and
options distinct; every 1.1 rule recomputed from the prompt (row_rule_ok); fwd is a topological order and rev its exact
reverse with the same item_id and options; K1 rows share the KH rows' statements and order, query the target chain's
hop-1 variable and show the hop-1 set (ca / wp) or the five roots (vb); the -qs twin has the same statements and options
with gold D1; blind rows are the first 18 fwd items of each cell without statements; train_kd starts with train_kh;
disjointness (prompt hash, value key) recomputed; md5 of every file and all D1-D4 numbers recomputed from the rows equal
build_report.json and pass the bars."""
import hashlib
import json
import os
import random
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_khop as B  # noqa: E402

KHOP = Path(os.environ.get("KHOP_EVAL_DIR") or ROOT.parents[1] / "src/data/eval/khop")
SMALL = {t: {1: 5, 2: 5, 3: 5} for t in ("ca", "wp", "vb")} | {"cag": {1: 5, 2: 5}}


def load(stem):
    return [json.loads(l) for l in open(KHOP / f"{stem}.jsonl")]


def row(prompt, gold="A", group="khop/ca/fwd/k2", depth=2):
    opts = prompt.split("\nOptions:\n")[1].split("\n")
    return {"prompt": prompt, "gold_label": gold, "group": group, "depth": depth,
            "options": [o.split(") ")[1] for o in opts]}


class TestGenerator(unittest.TestCase):
    def test_deterministic_by_seed(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
            ra = B.build(Path(a), n_kh=12, n_kd_extra=6, eval_counts=SMALL)
            rb = B.build(Path(b), n_kh=12, n_kd_extra=6, eval_counts=SMALL)
            rc = B.build(Path(c), n_kh=12, n_kd_extra=6, eval_counts=SMALL, eval_seed=3)
            self.assertEqual(ra["md5"], rb["md5"])
            self.assertEqual(ra["md5"]["train_kh.jsonl"], rc["md5"]["train_kh.jsonl"])
            self.assertNotEqual(ra["md5"]["khop_ca_fwd.jsonl"], rc["md5"]["khop_ca_fwd.jsonl"])

    def test_option_rule(self):
        self.assertTrue(B.one_op(10, 15, 2, 9) and B.one_op(10, 20, 2, 9) and B.one_op(10, 30, 2, 9))
        self.assertTrue(B.one_op(15, 10, 2, 9))                       # |g - h| covers + and - (one_op_parent line)
        self.assertFalse(B.one_op(10, 11, 2, 9) or B.one_op(10, 25, 2, 9) or B.one_op(20, 10, 2, 9))
        self.assertFalse(B.option_set_ok([10, 20, 34, 46, 58], set()))          # 20 = 2 x 10
        self.assertFalse(B.option_set_ok([11, 23, 33, 47, 59], set()))          # 33 = 3 x 11
        self.assertTrue(B.option_set_ok([11, 13, 15, 16, 18], set()))           # 2-9 apart is allowed
        self.assertFalse(B.option_set_ok([11, 23, 35, 47, 59], {35}))           # a written number
        self.assertFalse(B.option_set_ok([11, 11, 35, 47, 59], set()))
        stats = defaultdict(Counter)
        rng = random.Random(0)
        for task in ("ca", "wp", "cag"):
            vlo, vhi = B.RANGES[task][:2]
            for k in (1, 2, 4, 8):
                for _ in range(20):
                    ch = B.draw_arith_item(task, k, rng, stats[(task, k)])
                    self.assertEqual(len({vals[0] for vals, _ in ch}), 1)                    # one root value
                    self.assertEqual(len({tuple(o for o, _ in ops) for _, ops in ch}), 1)  # one op-kind sequence
                    self.assertIn(ch[0][1][0][0], ("+", "-"))                                # hop 1 is + or -
                    self.assertEqual(len({ops[0][1] for _, ops in ch}), 5)                 # five distinct hop-1 constants
                    self.assertTrue(all(vlo <= v <= vhi for vals, _ in ch for v in vals))
                    self.assertEqual(len({vals[k] for vals, _ in ch}), 5)
        self.assertLess(stats[("ca", 8)]["draws"] / stats[("ca", 8)]["items"], 20)      # far below MAX_RESAMPLE
        self.assertEqual(B.MAX_RESAMPLE, 10_000)

    def test_skeleton_rule_on_rows(self):
        ok = ("Question: bo = 7. ki = bo + 5. zu = 7. fa = zu + 3. re = 7. ti = re + 8. va = 7. du = va + 2. "
              "wu = 7. ne = wu + 6. What is the value of ki?\nOptions:\nA) 12\nB) 10\nC) 15\nD) 13\nE) 9")
        self.assertEqual(B.row_rule_ok(row(ok, group="khop/ca/fwd/k1", depth=1)), [])
        bad_root = ok.replace("zu = 7.", "zu = 8.").replace("B) 10", "B) 11")
        self.assertIn("roots differ (skeleton)", B.row_rule_ok(row(bad_root, group="khop/ca/fwd/k1", depth=1)))
        bad_c = ok.replace("fa = zu + 3.", "fa = zu + 5.").replace("B) 10", "B) 12")
        errs = B.row_rule_ok(row(bad_c, group="khop/ca/fwd/k1", depth=1))
        self.assertTrue(any("five distinct constants" in e for e in errs), errs)

    def test_interleave_is_a_linear_extension(self):
        rng = random.Random(1)
        first = Counter()
        for _ in range(2000):
            seq = B.interleave(3, rng)
            nxt = [0] * 5
            for c, j in seq:
                self.assertEqual(j, nxt[c])
                nxt[c] += 1
            first[seq[0][0]] += 1
        self.assertTrue(all(330 < first[c] < 470 for c in range(5)), first)    # each chain first about 1 / 5

    def test_parse_and_baselines(self):
        p = ("Question: bo = 7. ki = bo + 5. zu = 40. fa = zu × 2. What is the value of ki?\n"
             "Options:\nA) 12\nB) 80\nC) 3\nD) 41\nE) 66")
        P = B.parse_prompt(p)
        self.assertEqual(P["query"], "ki")
        self.assertEqual(P["values"], {"bo": 7, "ki": 12, "zu": 40, "fa": 80})
        self.assertEqual(P["literals"], [7, 5, 40, 2])
        b = B.baselines(row(p))
        self.assertEqual(b["max"], 0.0)
        self.assertEqual(b["first_lit"], 0.0)            # closest to 7: 3 (C)
        self.assertEqual(b["last_lit"], 0.0)              # closest to 2: 3 (C)
        self.assertEqual(b["closest_lit"], 0.0)           # 41 is 1 from 40, 3 is 1 from 2: C, D
        self.assertEqual(b["unwritten"], 0.2)             # five unwritten options
        self.assertEqual(b["qop_pair"], 0.2)              # no g = h + 5 among the options: uniform
        self.assertEqual(b["qop_consistent"], 0.25)       # + 5: options > 5 are A, B, D, E
        w = ("Question: Mia has 7 marbles. Leo has twice as many marbles as Mia. How many marbles does Leo have?\n"
             "Options:\nA) 14\nB) 30\nC) 51\nD) 66\nE) 85")
        Pw = B.parse_prompt(w)
        self.assertEqual(Pw["values"], {"Mia": 7, "Leo": 14})
        self.assertEqual(B.baselines(row(w, group="khop/wp/fwd/k1", depth=1))["qop_consistent"], 1 / 3)  # 14, 30, 66
        bl = "Question: What is the value of ki?\nOptions:\nA) 12\nB) 80\nC) 3\nD) 41\nE) 66"
        bb = B.baselines(row(bl, group="khop/ca/blind/k2"))
        self.assertEqual((bb["unwritten"], bb["qop_pair"], bb["closest_lit"]), (0.2, 0.2, 0.2))
        with self.assertRaises(ValueError):
            B.parse_prompt("Question: bo is 7. What is the value of bo?\nOptions:\nA) 7")

    def test_chase_lines_and_hard(self):
        p = ("Question: bo = 7. ki = bo + 5. me = ki × 2. zu = 7. fa = zu + 3. ru = fa × 2. What is the value of me?\n"
             "Options:\nA) 24\nB) 20\nC) 30\nD) 34\nE) 40")
        b = B.baselines(row(p))
        self.assertEqual(B.query_hops(B.parse_prompt(p)), (7, [("+", 5), ("*2", 2)]))
        self.assertEqual(b["chase_mult"], 0.0)            # 7 x 2 = 14: nearest 20 (B)
        self.assertEqual(b["chase_sign"], 1.0)            # (7 + 5.5) x 2 = 25: nearest 24 (A)
        self.assertEqual(b["chase_droplast"], 1.0)        # hop 1 exact, the x2 hop has no constant: 24
        self.assertEqual(b["drop_one_best"], 1.0)         # the only additive hop dropped: 25 -> A
        self.assertEqual(b["hard"], 0.0)                  # a drop-one estimate finds the gold
        self.assertEqual(b["qop_avoid"], 0.25)            # x2 pair: 40 = 2 x 20 (E); uniform over A-D
        self.assertTrue(set(B.ALL_LINES) <= set(b))
        self.assertIs(B.hard_item(row(p)), False)
        h = p.replace("A) 24", "A) 26").replace("B) 20", "B) 24").replace("ki = bo + 5.", "ki = bo + 6.")   # gold 26
        self.assertEqual(B.baselines(row(h))["chase_sign"], 0.5)                  # 25: 26 (A) and 24 (B) tie
        vb = "Question: bo = 47. ki = bo. What is the value of ki?\nOptions:\nA) 47\nB) 20\nC) 30\nD) 34\nE) 40"
        bv = B.baselines(row(vb, group="khop/vb/fwd/k1", depth=1))
        self.assertEqual((bv["chase_sign"], bv["chase_mult"], bv["hard"]), (0.2, 0.2, None))
        self.assertIsNone(B.hard_item(row(vb, group="khop/vb/fwd/k1", depth=1)))

    def test_one_op_parent_value(self):
        p = ("Question: bo = 7. ki = bo + 5. zu = 40. fa = zu × 2. What is the value of ki?\n"
             "Options:\nA) 12\nB) 80\nC) 3\nD) 41\nE) 66")
        # candidates: 12 (3 + 9), 3 (12 - 9)... and 80 / 41 / 66 have no one-op partner; gold A is one of 2 -> 0.5
        self.assertEqual(B.baselines(row(p))["one_op_parent"], 0.5)


@unittest.skipUnless((KHOP / "build_report.json").exists(), f"no built khop files in {KHOP}")
class TestRealFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = json.loads((KHOP / "build_report.json").read_text())
        cls.f = {s: load(s) for s in B.EVAL_FILES + B.TRAIN_FILES}

    def test_md5_and_counts(self):
        for s in self.f:
            self.assertEqual(hashlib.md5((KHOP / f"{s}.jsonl").read_bytes()).hexdigest(), self.rep["md5"][f"{s}.jsonl"], s)
        for s in ("khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_wp_rev", "khop_vb_fwd", "khop_vb_rev"):
            c = Counter(r["k"] for r in self.f[s])
            self.assertEqual(c, Counter({k: 150 for k in range(1, 7)} | {7: 100, 8: 100}), s)
        self.assertEqual(Counter(r["k"] for r in self.f["khop_cag_fwd"]), Counter({k: 150 for k in range(1, 7)}))
        self.assertEqual(len(self.f["khop_ca_fwd_qs"]), 900)
        self.assertEqual(Counter((r["task"], r["k"]) for r in self.f["khop_blind"]),
                         Counter({(t, k): 18 for t in B.TASKS for k in range(1, 7)}))
        self.assertEqual([len(self.f[s]) for s in B.TRAIN_FILES], [3000, 3000, 9000])
        kh = self.f["train_kh"]
        self.assertEqual(Counter(B.task_of(r) for r in kh), Counter({t: 1000 for t in B.TASKS}))
        orders = Counter((B.task_of(r), r["group"].split("/")[2]) for r in kh)   # alternating blocks of 6 rows: 502 / 498
        self.assertEqual(orders, Counter({(t, o): (502 if o == "fwd" else 498) for t in B.TASKS for o in ("fwd", "rev")}))
        self.assertTrue(all(c in (166, 167) for c in Counter((B.task_of(r), r["depth"]) for r in kh).values()))

    def test_keys_and_basic_fields(self):
        for s, rows in self.f.items():
            want = set(B.GSMVT_KEYS) | (set() if s.startswith("train") else set(B.EVAL_EXTRA))
            ids = set()
            for r in rows:
                self.assertEqual(set(r), want, s)
                self.assertIn(r["gold_label"], r["labels"])
                self.assertEqual(r["teacher"], {L: float(L == r["gold_label"]) for L in B.LETTERS})
                self.assertEqual(r["option_kinds"][B.LETTERS.index(r["gold_label"])], "gold")
                self.assertEqual(r["option_kinds"].count("cross"), 4)
                self.assertEqual(len(set(r["options"])), 5)
                self.assertEqual(r["source"], "khop")
                self.assertEqual((r["depth"], r["n_steps"], r["dep_level"]), (r["depth"],) * 3)
                ids.add(r["item_id"])
            self.assertEqual(len(ids), len(rows), s)
        gsm = ROOT.parents[1] / "src/data/student_gsm8k_tf_vt/train.jsonl"
        if gsm.exists():
            with open(gsm) as fh:
                self.assertEqual(set(json.loads(fh.readline())), set(B.GSMVT_KEYS))

    def test_rules_every_row(self):
        for s, rows in self.f.items():
            for r in rows:
                self.assertEqual(B.row_rule_ok(r), [], (s, r["item_id"]))

    def test_orders(self):
        for t in B.TASKS:
            fw, rv = self.f[f"khop_{t}_fwd"], self.f[f"khop_{t}_rev"]
            self.assertEqual([r["item_id"] for r in fw], [r["item_id"] for r in rv])
            for a, b in zip(fw, rv):
                self.assertTrue(B.fwd_is_topological(a["prompt"]))
                self.assertFalse(b["depth"] > 0 and B.fwd_is_topological(b["prompt"]))
                sa = [s["name"] for s in B.parse_prompt(a["prompt"])["stmts"]]
                sb = [s["name"] for s in B.parse_prompt(b["prompt"])["stmts"]]
                self.assertEqual(sa, sb[::-1])
                self.assertEqual((a["options"], a["gold_label"]), (b["options"], b["gold_label"]))
                self.assertEqual((a["order"], b["order"]), ("fwd", "rev"))
        for r in self.f["train_kh"]:
            o = r["group"].split("/")[2]
            self.assertEqual(B.fwd_is_topological(r["prompt"]), o == "fwd", r["item_id"])

    def test_k1_qs_blind_kd(self):
        kh = {r["item_id"]: r for r in self.f["train_kh"]}
        for r in self.f["train_k1"]:
            base = kh[r["item_id"][:-3]]
            self.assertTrue(r["item_id"].endswith("-k1"))
            Pa, Pb = B.parse_prompt(base["prompt"]), B.parse_prompt(r["prompt"])
            self.assertEqual(Pa["stmts"], Pb["stmts"])
            self.assertEqual(base["group"], r["group"])
            roots = [s["name"] for s in Pa["stmts"] if s["op"] == "root"]
            tgt_root = Pa["chain_of"][Pa["query"]]
            hop1 = {s["parent"]: s["name"] for s in Pa["stmts"] if s["parent"]}
            self.assertEqual(Pb["query"], hop1[tgt_root])                         # target chain's hop-1 variable
            want = sorted(Pa["values"][hop1[x]] for x in roots)
            self.assertEqual(sorted(Pb["options"]), want)                          # hop-1 set (vb: the roots)
        ca = {r["item_id"]: r for r in self.f["khop_ca_fwd"]}
        for q in self.f["khop_ca_fwd_qs"]:
            o = ca[q["item_id"][:-3]]
            Po, Pq = B.parse_prompt(o["prompt"]), B.parse_prompt(q["prompt"])
            self.assertEqual(Po["stmts"], Pq["stmts"])
            self.assertEqual(o["options"], q["options"])
            self.assertNotEqual(Po["chain_of"][Po["query"]], Pq["chain_of"][Pq["query"]])
            self.assertEqual(o["option_kinds"][B.LETTERS.index(q["gold_label"])], "cross")
        firsts = defaultdict(list)
        for t in B.TASKS:
            for r in self.f[f"khop_{t}_fwd"]:
                if r["k"] <= 6 and len(firsts[(t, r["k"])]) < 18:
                    firsts[(t, r["k"])].append(r)
        for b in self.f["khop_blind"]:
            src = next(r for r in firsts[(b["task"], b["k"])] if r["item_id"] == b["item_id"])
            self.assertEqual((b["options"], b["gold_label"]), (src["options"], src["gold_label"]))
            self.assertEqual(B.parse_prompt(b["prompt"])["stmts"], [])
            self.assertEqual(B.parse_prompt(b["prompt"])["query"], B.parse_prompt(src["prompt"])["query"])
        self.assertEqual(self.f["train_kd"][:3000], self.f["train_kh"])

    def test_d_lines_recomputed(self):
        rep = B.d_lines(self.f, None)
        self.assertEqual(rep["d1"], self.rep["d1_baselines"])
        self.assertEqual(rep["d2"], self.rep["gold_shares"])
        self.assertEqual(rep["fail"], [x for x in self.rep["d_fail"] if not x.startswith(("D3", "D4"))])
        self.assertEqual(self.rep["d_fail"], [])
        self.assertTrue(self.rep["d_pass"])
        d3 = self.rep["d3_disjointness"]
        self.assertEqual((d3["shared_prompt_hashes"], d3["shared_value_keys"]), (0, 0))
        tr_h = {B.phash(r["prompt"]) for s in B.TRAIN_FILES for r in self.f[s]}
        ev_h = {B.phash(r["prompt"]) for s in B.EVAL_FILES for r in self.f[s]}
        self.assertFalse(tr_h & ev_h)
        if self.rep.get("d4_tokens"):
            mx = max(c["max"] for s in self.rep["d4_tokens"].values() for c in s.values())
            self.assertLess(mx, 1536)

    def test_d4_recomputed_with_qwen(self):
        if not self.rep.get("tokenizer"):
            self.skipTest("report built without a tokenizer")
        try:
            import transformers
            tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
        except Exception as e:
            self.skipTest(f"no local Qwen3.5-4B tokenizer: {e}")
        for s in ("khop_wp_fwd", "train_kd"):
            got = defaultdict(list)
            for r in self.f[s]:
                got[f"{B.task_of(r)}/k{r['depth']}"].append(B.n_tokens(tok, r["prompt"]))
            self.assertEqual({c: max(v) for c, v in got.items()},
                             {c: v["max"] for c, v in self.rep["d4_tokens"][s].items()})


if __name__ == "__main__":
    unittest.main()
