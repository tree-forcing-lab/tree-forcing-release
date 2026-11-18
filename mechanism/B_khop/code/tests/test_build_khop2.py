"""scripts/build_khop2.py (CARD_khop_v2 sections 1.2, 2.5 D1-D6, 5) and one A2 step on a 3-copy extra file.
Generator level (no data needed): the copy rule (copy 1 byte for byte, copy c with item_id + "-r<c>" and nothing else
changed, a reverted copy re-serializes to the parent line), copies_ok refuses a changed field, row_key (parsed from the
prompt) equals build_khop.value_key on generated items.
Small builds against the real parent files (E/src/data/eval/khop or KHOP_EVAL_DIR; skipped when absent): determinism by
seed, the parent files are not touched (md5 before and after), a changed parent md5 is refused, --out = --src is refused.
The real files (E/src/data/eval/khop2 or KHOP2_EVAL_DIR; skipped when absent): md5 and counts equal build_report.json;
train_kd27 starts with train_kd byte for byte; the 18,000 new rows follow the k / order / id / gold-layout rules of 1.2,
have the 32 gsmvt keys, pass every 1.1 rule (build_khop.row_rule_ok) and share no value key or prompt hash with the eval
files, train_kh, train_k1, train_kd or another new row; the copy files are 3 x their parent apart from the item_id
suffixes; all item_ids distinct; D1 / D2 / D3 / D6 recomputed from the rows equal the report and pass the bars; D4
recomputed with the Qwen3.5-4B tokenizer (when cached).
A2 step (torch + transformers; tests/test_extra_train.py pattern): 1-rank FSDP (torchrun --local-addr 127.0.0.1,
cpu-adamw), GSM rows + a 3-copy khop file, --eval-value-render vt: runs end to end and logs the extra-train line with
3 x the base row count."""
import hashlib
import json
import os
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
sys.path.insert(0, str(ROOT / "src"))
import build_khop as B  # noqa: E402
import build_khop2 as K  # noqa: E402

KHOP = Path(os.environ.get("KHOP_EVAL_DIR") or ROOT.parents[1] / "src/data/eval/khop")
KHOP2 = Path(os.environ.get("KHOP2_EVAL_DIR") or ROOT.parents[1] / "src/data/eval/khop2")
HAVE_PARENT = (KHOP / "build_report.json").exists()
HAVE_REAL = (KHOP2 / "build_report.json").exists() and HAVE_PARENT
# CARD_khop_v1 Amendment 2 table (E/src_khop/BUILD_REPORT.md, "Data (rebuilt 2026-10-26 00:22 CDT)")
PARENT_MD5 = {"khop_ca_fwd": "171f097c2f0f55ae4505d37365d95491", "khop_ca_rev": "2f2bc7b6fc1cfd45f24451741f0e4105",
              "khop_wp_fwd": "c58c93ffa4d3912cfdbfd848f6eb9fb9", "khop_wp_rev": "bdad8780cc64e94e3fcdd0148833a4d7",
              "khop_vb_fwd": "046c6ad382d66c02f1315ce7f8c7577d", "khop_vb_rev": "2279c09f4adb666196bdbbe636b53646",
              "khop_cag_fwd": "3e79c1b5233e3db7dc23927e3066eba5", "khop_ca_fwd_qs": "38cf266ea1490f0c42f6378baa85f4e4",
              "khop_blind": "b58f7284e1b1d88197a44d0b54eba2ae", "train_kh": "a16b1520f0fb8b24ffd8fa57ef696de0",
              "train_k1": "4bb15f66c8d8843013eb0410c6a87e48", "train_kd": "2ed4ecc377e4638e35b12e8b22e37461"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def small_rows(n=4, seed=1):
    kh, _, _, _, _ = B.build_train(__import__("random").Random(seed), defaultdict(Counter), n, 0)
    return kh


class TestCopyRule(unittest.TestCase):
    def test_copies_and_revert(self):
        rows = small_rows()
        lines = [K.dump(r) for r in rows]
        out = K.copies(lines, 3)
        n = len(lines)
        self.assertEqual(len(out), 3 * n)
        self.assertEqual(out[:n], lines)                                   # copy 1 byte for byte
        for c in (2, 3):
            for a, b in zip(lines, out[(c - 1) * n:c * n]):
                ra, rb = json.loads(a), json.loads(b)
                self.assertEqual(rb["item_id"], ra["item_id"] + f"-r{c}")
                self.assertEqual(rb["problem_id"], ra["problem_id"])        # problem_id unchanged
                self.assertEqual(K.dump({**rb, "item_id": ra["item_id"]}), a)   # D6: only the item_id differs
        self.assertTrue(K.copies_ok(out, lines, 3))
        self.assertFalse(K.copies_ok(out, lines, 2))
        bad = list(out)
        r = json.loads(bad[n + 1])
        r["gold_label"] = "E" if r["gold_label"] != "E" else "A"
        bad[n + 1] = K.dump(r)
        self.assertFalse(K.copies_ok(bad, lines, 3))                       # another field changed
        bad = list(out)
        bad[0] = bad[0].replace(b'"source": "khop"', b'"source":  "khop"')
        self.assertFalse(K.copies_ok(bad, lines, 3))                       # copy 1 not byte for byte
        self.assertEqual(len({json.loads(l)["item_id"] for l in out}), 3 * n)

    def test_row_key_equals_value_key(self):
        import random
        rng, stats = random.Random(5), defaultdict(Counter)
        for task in B.TASKS:
            for k in (1, 2, 3, 6):
                it = B.gen_item(task, k, rng, stats)
                r = it.row("x", "rev", B.layout(0, k, rng))
                self.assertEqual(K.row_key(r), B.value_key(task, it.ch))
        blind = it.row("x", "fwd", B.layout(0, k, rng), blind=True)
        self.assertIsNone(K.row_key(blind))


@unittest.skipUnless(HAVE_PARENT, f"no parent khop files in {KHOP}")
class TestSmallBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.before = {s: md5(KHOP / f"{s}.jsonl") for s in K.PARENT_FILES}
        cls.a = K.build(KHOP, cls.d / "a", seed=3, n_per_task=12)
        cls.b = K.build(KHOP, cls.d / "b", seed=3, n_per_task=12)
        cls.c = K.build(KHOP, cls.d / "c", seed=4, n_per_task=12)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_deterministic_by_seed(self):
        self.assertEqual(self.a["md5"], self.b["md5"])
        self.assertNotEqual(self.a["md5"]["train_kd27.jsonl"], self.c["md5"]["train_kd27.jsonl"])
        self.assertEqual(self.a["md5"]["train_kd_x3.jsonl"], self.c["md5"]["train_kd_x3.jsonl"])   # no rng in copies
        self.assertEqual(self.a["counts"]["train_kd27"]["total"], 9000 + 36)

    def test_parent_untouched(self):
        self.assertEqual({s: md5(KHOP / f"{s}.jsonl") for s in K.PARENT_FILES}, self.before)
        self.assertEqual({s: self.a["parent_md5"][f"{s}.jsonl"] for s in K.PARENT_FILES}, PARENT_MD5)

    def test_refuses_changed_parent_and_same_dir(self):
        src = self.d / "src"
        shutil.copytree(KHOP, src)
        with open(src / "train_k1.jsonl", "ab") as f:
            f.write(b"\n")
        with self.assertRaises(SystemExit) as cm:
            K.build(src, self.d / "x", n_per_task=6)
        self.assertIn("md5", str(cm.exception))
        self.assertFalse((self.d / "x" / "train_kd27.jsonl").exists())
        with self.assertRaises(SystemExit):
            K.build(KHOP, KHOP, n_per_task=6)

    def test_redraw_cap_raises(self):
        orig = K.MAX_REDRAW
        try:
            K.MAX_REDRAW = 0
            forbid = set()
            import random
            rng, st = random.Random(3), defaultdict(Counter)
            for _ in range(200):                                      # forbid the first draws of ca k = 1
                forbid.add(B.value_key("ca", B.gen_item("ca", 1, rng, st).ch))
            with self.assertRaises(B.BuildError):
                K.new_rows(3, 1, forbid, set(), Counter())
        finally:
            K.MAX_REDRAW = orig


@unittest.skipUnless(HAVE_REAL, f"no built khop2 files in {KHOP2}")
class TestRealFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = json.loads((KHOP2 / "build_report.json").read_text())
        cls.raw = {s: (KHOP2 / f"{s}.jsonl").read_bytes().splitlines(keepends=True) for s in K.NEW_FILES}
        cls.f = {s: [json.loads(l) for l in v] for s, v in cls.raw.items()}
        cls.praw = {s: (KHOP / f"{s}.jsonl").read_bytes().splitlines(keepends=True) for s in K.PARENT_FILES}
        cls.p = {s: [json.loads(l) for l in v] for s, v in cls.praw.items()}
        cls.new = cls.f["train_kd27"][9000:]

    def test_md5_and_counts(self):
        for s in K.NEW_FILES:
            self.assertEqual(md5(KHOP2 / f"{s}.jsonl"), self.rep["md5"][f"{s}.jsonl"], s)
        self.assertEqual({s: md5(KHOP / f"{s}.jsonl") for s in K.PARENT_FILES}, PARENT_MD5)
        self.assertEqual({k[:-6]: v for k, v in self.rep["parent_md5"].items()}, PARENT_MD5)
        self.assertEqual({s: len(v) for s, v in self.f.items()}, {"train_kd27": 27000, "train_kd_x3": 27000,
                                                                "train_kh_x3": 9000})
        self.assertEqual(self.rep["counts"]["train_kd27"], {f"{t}/k{k}": 1500 for t in B.TASKS for k in range(1, 7)} |
                         {"total": 27000})
        self.assertEqual(self.rep["counts_order"]["train_kd27_new"],
                         {f"{t}/{o}/k{k}": 500 for t in B.TASKS for o in ("fwd", "rev") for k in range(1, 7)})
        self.assertEqual((self.rep["seed"], self.rep["card"], self.rep["repeat"]), (3, "CARD_khop_v2", 3))

    def test_nested_and_new_row_rules(self):
        self.assertEqual(self.raw["train_kd27"][:9000], self.praw["train_kd"])
        self.assertEqual(len(self.new), 18000)
        cyc = Counter()
        for n, r in enumerate(self.new):
            task = B.TASKS[n // 6000]
            j = 3000 + n % 6000
            k = 1 + j % 6
            order = "fwd" if (j // 6) % 2 == 0 else "rev"
            self.assertEqual((r["item_id"], r["depth"], r["group"]), (f"khop-tr-{task}-{j:05d}", k,
                                                                      f"khop/{task}/{order}/k{k}"))
            self.assertEqual(set(r), set(B.GSMVT_KEYS))
            i = 500 + cyc[(task, k)]
            cyc[(task, k)] += 1
            self.assertEqual(r["gold_label"], B.LETTERS[(i + 3 * (k - 1)) % 5])     # layout(i) with i from 500
            self.assertEqual(r["problem_id"], r["item_id"])
        bad = [r["item_id"] for r in self.new if B.row_rule_ok(r)]
        self.assertEqual(bad, [])
        self.assertTrue(all(B.fwd_is_topological(r["prompt"]) for r in self.new if "/fwd/" in r["group"]))

    def test_disjoint(self):
        keys = [K.row_key(r) for r in self.new]
        hs = [B.phash(r["prompt"]) for r in self.new]
        self.assertEqual((len(set(keys)), len(set(hs))), (18000, 18000))       # no repeat among the new rows
        pk = {K.row_key(r) for s in K.PARENT_FILES for r in self.p[s]} - {None}
        ph = {B.phash(r["prompt"]) for s in K.PARENT_FILES for r in self.p[s]}
        self.assertFalse(set(keys) & pk)
        self.assertFalse(set(hs) & ph)
        ek = {K.row_key(r) for s in B.EVAL_FILES for r in self.p[s]} - {None}
        eh = {B.phash(r["prompt"]) for s in B.EVAL_FILES for r in self.p[s]}
        for s in K.NEW_FILES:
            d = self.rep["d3_disjointness"][s]
            got = {"shared_prompt_hashes": len({B.phash(r["prompt"]) for r in self.f[s]} & eh),
                   "shared_value_keys": len({K.row_key(r) for r in self.f[s]} & ek)}
            self.assertEqual(got, d, s)
            self.assertEqual(got, {"shared_prompt_hashes": 0, "shared_value_keys": 0})
        self.assertEqual(self.rep["new_distinct_keys"], 18000)
        self.assertEqual(self.rep["distinct_value_keys"]["train_kd27"], 26941)
        self.assertEqual(len({K.row_key(r) for r in self.f["train_kd27"]}), 26941)

    def test_copies(self):
        self.assertTrue(K.copies_ok(self.raw["train_kd_x3"], self.praw["train_kd"], 3))
        self.assertTrue(K.copies_ok(self.raw["train_kh_x3"], self.praw["train_kh"], 3))
        self.assertEqual(self.rep["copies_ok"], {"train_kd_x3": True, "train_kh_x3": True})
        self.assertTrue(self.rep["nested"])
        for s in K.NEW_FILES:
            ids = [r["item_id"] for r in self.f[s]]
            self.assertEqual(len(set(ids)), len(ids), s)
        self.assertTrue(self.rep["item_ids_distinct"])

    def test_d_lines_recomputed(self):
        files = dict(self.f, train_kd27_new=self.new)
        rep = B.d_lines(files, None)
        self.assertEqual(rep["d1"], self.rep["d1_baselines"])
        self.assertEqual(rep["d2"], self.rep["gold_shares"])
        self.assertEqual(rep["fail"], [])
        self.assertEqual(self.rep["d_fail"], [])
        self.assertTrue(self.rep["d_pass"])
        self.assertEqual(set(self.rep["d1_baselines"]), set(K.D_STEMS))
        for s in K.D_STEMS:                                  # the bars, read from the report itself
            for c, v in self.rep["d1_baselines"][s].items():
                bar = B.BARS["pooled"] if c.startswith("pooled") else B.BARS["cell"]
                self.assertTrue(all(v[b] <= bar for b in B.barred(s)), (s, c))
                if not c.startswith("pooled"):
                    self.assertGreaterEqual(v["n"], 100)
            for c, g in self.rep["gold_shares"][s].items():
                self.assertTrue(all(B.BARS["gold_lo"] <= x <= B.BARS["gold_hi"] for x in g.values()), (s, c))
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
        got = defaultdict(list)
        for r in self.new:
            got[f"{B.task_of(r)}/k{r['depth']}"].append(B.n_tokens(tok, r["prompt"]))
        self.assertEqual({c: max(v) for c, v in got.items()},
                         {c: v["max"] for c, v in self.rep["d4_tokens"]["train_kd27_new"].items()})
        self.assertEqual(sum(sum(v) for v in got.values()), self.rep["train_tokens"]["train_kd27_new"])


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from test_depth_forcing import gpt2_tok, tiny_qwen35
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestA2StepThreeCopies(unittest.TestCase):
    def test_fsdp_one_rank_three_copies(self):
        try:
            tok = gpt2_tok()
        except Exception as e:
            self.skipTest(f"no local gpt2 tokenizer: {e}")
        with tempfile.TemporaryDirectory() as d:
            tiny_qwen35(len(tok)).save_pretrained(d + "/m")
            tok.save_pretrained(d + "/m")
            gsm = json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())["gsm8k"]
            Path(d, "gsm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gsm))
            base = [K.dump(r) for r in small_rows(2)]
            Path(d, "x3.jsonl").write_bytes(b"".join(K.copies(base, 3)))
            Path(d, "ev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gsm[:3] + small_rows(1)))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
            x3, out = d + "/x3.jsonl", d + "/out"
            r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
                                "--local-addr", "127.0.0.1", str(ROOT / "scripts/train_student.py"), "--model", d + "/m",
                                "--train", d + "/gsm.jsonl", "--extra-train", x3, "--eval", "val=" + d + "/ev.jsonl",
                                "--eval-value-render", "vt", "--final", "teacher", "--subq", "none", "--epochs", "1",
                                "--micro-bs", "2", "--grad-accum", "2", "--max-len", "512", "--eval-max-len", "512",
                                "--precision", "fp32master", "--lr", "1e-3", "--log-every", "1", "--seed", "1", "--fsdp",
                                "--fsdp-optim", "cpu-adamw", "--label-smoothing", "0.1", "--out", out],
                               capture_output=True, text=True, timeout=900, env=env)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            n = 3 * len(base)
            self.assertIn(f"extra-train: {n} rows from {x3} (sources {{'khop': {n}}})", r.stdout)
            self.assertIn(f"extra-train: total training rows {len(gsm) + n} ({len(gsm)} from {d}/gsm.jsonl)", r.stdout)
            m = json.loads(Path(out, "metrics.json").read_text())
            self.assertEqual(m["extra_train"], {x3: n})
            self.assertEqual(m["fsdp"]["world"], 1)
            steps = -(-(len(gsm) + n) // 4)                       # ceil(rows / (micro-bs x grad-accum))
            self.assertEqual(m["history"][-1]["step"], steps)
            self.assertTrue(all(x["loss"] == x["loss"] for x in m["history"]))


if __name__ == "__main__":
    unittest.main()
