"""scripts/build_khop4.py and the two stage-2 protections on the tiny model.
Real-file tests (KHOP4_DATA_ROOT or E/src/data: student_khop3_*, eval/khop, eval/khop3, student_gsm8k_tf_vt):
- the 4 outputs equal the khop3 inputs byte for byte; train_kh = rows 0..2,999 of student_khop4_c27/train.jsonl and
  train_k1 = rows 0..2,999 of student_khop4_k1x/train.jsonl byte for byte (D2); the report values recomputed from the
  files equal the report; two builds are byte-equal and equal to the files and the report in the data root; the inputs
  are unchanged (md5 before and after, against the design values); the replay item_ids (khop-tr-*) and the gsmvt train
  item_ids are disjoint (the trainer stops on a duplicate, so R would not start otherwise).
- refusals before any write: a changed row in each of the 6 inputs, a parent report with another md5, a row count off,
  a non-object report, --out-root / --report / an output dir equal to an input dir, an output dir or report dir that
  holds another file.
- the tiny two-stage run (tests/test_build_khop3.py pattern: torchrun --local-addr 127.0.0.1, FSDP cpu-adamw,
  --eval-value-render vt, label smoothing 0.1, --token-budget 4096, seed = order seed 16, micro-bs 2, grad-accum 2):
  stage 1 with --save on 18 khop rows, then from the saved weights on 6 GSM rows:
  (a) arm L: --lr at one quarter of the reference (2.5e-4 against 1e-3): the history lr values are exactly one quarter
      of the reference run's at every step (--log-every 1; a power-of-two factor is exact in floating point), the
      per-update examples (item_id, text, labels, target) equal the reference's, losses finite, final step 2;
  (b) arm R: --extra-train of 6 khop rows (disjoint from the stage-1 rows): log lines "extra-train: 6 rows from <path>"
      and "extra-train: total training rows 12 (6 from <gsm path>)", "stage all: 12 examples per pass, 3 steps", final
      step ceil((6 + 6) / 4) = 3, metrics.json extra_train {<path>: 6}, finite losses, every training example is a GSM
      or a replay row and every one of the 12 rows is seen; a --extra-train file that repeats a training item_id stops
      the trainer (SystemExit, nonzero exit) before any update.
Run: python -m unittest tests.test_build_khop4 (from the repo)."""
import hashlib
import json
import math
import os
import random
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
import build_khop4 as K4  # noqa: E402

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from test_depth_forcing import gpt2_tok, tiny_qwen35
    import test_build_khop3 as K3T          # TRACER, TORCHRUN, ENV (module import: its tests are not collected here)
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

E_DEFAULT = ROOT.parents[1]                     # E (repo = E/src_khop4/repo)
DATA = Path(os.environ.get("KHOP4_DATA_ROOT") or E_DEFAULT / "src/data")
HAVE_DATA = all((DATA / r).is_file() for r in K4.CARD_MD5) and (DATA / "eval/khop3/build_report.json").is_file()
TRAIN = str(ROOT / "scripts/train_student.py")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def copy_inputs(dst: Path):
    """The 6 inputs and the 2 parent reports under dst (a fresh data root)."""
    for rel in list(K4.CARD_MD5) + ["eval/khop3/build_report.json", "eval/khop/build_report.json"]:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / rel, dst / rel)
    return dst


@unittest.skipUnless(HAVE_DATA, "needs the khop3 data files")
class TestRealBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.before = {rel: md5(DATA / rel) for rel in K4.CARD_MD5}
        cls.rep = K4.build(DATA, d / "o1", d / "o1/eval/khop4")
        cls.rep2 = K4.build(DATA, d / "o2", d / "o2/eval/khop4")
        cls.d = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_passes(self):
        self.assertTrue(self.rep["d_pass"], self.rep["d_fail"])
        self.assertEqual(self.rep["d_fail"], [])
        self.assertTrue(all(self.rep["replay"].values()))
        self.assertEqual(len(self.rep["replay"]), 4)
        self.assertTrue(self.rep["inputs_unchanged"])

    def test_byte_copies(self):
        for k in K4.SRC:
            for f in K4.FILES:
                self.assertEqual((self.d / "o1" / K4.DST[k] / f).read_bytes(), (DATA / K4.SRC[k] / f).read_bytes())
            self.assertEqual(sorted(os.listdir(self.d / "o1" / K4.DST[k])), ["train.jsonl", "val.jsonl"])

    def test_d2_replay_rows(self):
        for rp, dd in (("eval/khop/train_kh.jsonl", "student_khop4_c27"), ("eval/khop/train_k1.jsonl", "student_khop4_k1x")):
            head = b"".join((self.d / "o1" / dd / "train.jsonl").read_bytes().splitlines(keepends=True)[:3000])
            self.assertEqual(head, (DATA / rp).read_bytes(), rp)
            self.assertEqual(hashlib.md5(head).hexdigest(), K4.CARD_MD5[rp])

    def test_report_recomputed(self):
        r = json.loads((self.d / "o1/eval/khop4/build_report.json").read_text())
        self.assertEqual(r["card"], "CARD_khop_v4")
        for k in K4.SRC:
            for f in K4.FILES:
                p = self.d / "o1" / K4.DST[k] / f
                rel = f"{K4.DST[k]}/{f}"
                self.assertEqual(r["md5"][rel], md5(p))
                self.assertEqual(r["rows"][rel], len(p.read_bytes().splitlines()))
        self.assertEqual(r["md5"]["student_khop4_c27/train.jsonl"], "92545dd42ca451c39e0e7c104aa78d1e")
        self.assertEqual(r["md5"]["student_khop4_k1x/train.jsonl"], "d7cc9716e5785461d16ec92156104261")
        self.assertEqual(r["md5"]["student_khop4_c27/val.jsonl"], "9f55fec541246e41401a7b5fa6c5510a")
        self.assertEqual({k: v["md5"] for k, v in r["inputs"].items()}, K4.CARD_MD5)
        self.assertEqual({k: v["rows"] for k, v in r["inputs"].items()}, K4.CARD_ROWS)
        self.assertEqual(r["file_sets"], {d: ["train.jsonl", "val.jsonl"] for d in K4.DST.values()})

    def test_deterministic_and_equal_to_data_root(self):
        a, b = self.d / "o1", self.d / "o2"
        for rel in [f"{d}/{f}" for d in K4.DST.values() for f in K4.FILES] + ["eval/khop4/build_report.json"]:
            self.assertEqual((a / rel).read_bytes(), (b / rel).read_bytes(), rel)
            if (DATA / rel).exists():
                self.assertEqual((a / rel).read_bytes(), (DATA / rel).read_bytes(), rel)

    def test_inputs_unchanged(self):
        self.assertEqual(self.before, K4.CARD_MD5)
        self.assertEqual({rel: md5(DATA / rel) for rel in K4.CARD_MD5}, K4.CARD_MD5)

    def test_replay_ids_disjoint_from_gsm(self):
        gsm = DATA / "student_gsm8k_tf_vt/train.jsonl"
        if not gsm.is_file():
            self.skipTest("no gsmvt train file")
        g = {json.loads(l)["item_id"] for l in gsm.read_text().split("\n") if l.strip()}
        self.assertEqual(len(g), 5923)
        for rp in ("eval/khop/train_kh.jsonl", "eval/khop/train_k1.jsonl"):
            ids = [json.loads(l)["item_id"] for l in (DATA / rp).read_text().split("\n") if l.strip()]
            self.assertEqual(len(set(ids)), 3000, rp)
            self.assertFalse(set(ids) & g, rp)
            self.assertTrue(all(i.startswith("khop-tr-") for i in ids), rp)


@unittest.skipUnless(HAVE_DATA, "needs the khop3 data files")
class TestRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.root = copy_inputs(self.d / "in")

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, out_root=None, report=None, want=""):
        out_root = out_root or self.d / "out"
        report = report or self.d / "rep"
        rp = report / "build_report.json"
        pre = rp.read_bytes() if rp.exists() else None
        with self.assertRaises(SystemExit) as c:
            K4.build(self.root, out_root, report)
        self.assertIn(want, str(c.exception))
        for dd in K4.DST.values():                                  # nothing written
            self.assertFalse((out_root / dd / "train.jsonl").exists(), out_root / dd)
            self.assertFalse((out_root / dd / "val.jsonl").exists(), out_root / dd)
        self.assertEqual(rp.read_bytes() if rp.exists() else None, pre, report)      # no report written
        return str(c.exception)

    def test_good_copy_builds(self):
        rep = K4.build(self.root, self.d / "out", self.d / "rep")
        self.assertTrue(rep["d_pass"])

    def test_changed_inputs(self):
        for rel in K4.CARD_MD5:
            p = self.root / rel
            orig = p.read_bytes()
            lines = orig.splitlines(keepends=True)
            lines[1] = lines[1].replace(b'"', b"'", 1)            # one changed row, same row count
            p.write_bytes(b"".join(lines))
            self.refused(want="md5")
            p.write_bytes(orig)

    def test_report_md5_and_rows(self):
        p = self.root / "eval/khop3/build_report.json"
        r = json.loads(p.read_text())
        r["md5"]["student_khop3_k1x/train.jsonl"] = "0" * 32
        p.write_text(json.dumps(r))
        self.refused(want="report")
        p.write_text("[1, 2]")
        self.refused(want="not a JSON object")
        p.write_text("{")
        self.refused(want="does not parse")
        shutil.copyfile(DATA / "eval/khop3/build_report.json", p)
        q = self.root / "eval/khop/build_report.json"
        r = json.loads(q.read_text())
        r["md5"]["train_kh.jsonl"] = "1" * 32
        q.write_text(json.dumps(r))
        self.refused(want="train_kh")
        shutil.copyfile(DATA / "eval/khop/build_report.json", q)
        os.remove(self.root / "eval/khop/train_k1.jsonl")
        self.refused(want="missing")

    def test_row_count(self):
        # a file with the design md5 cannot be short; the row check is the rows rule: no newline at the end stops
        p = self.d / "x.jsonl"
        p.write_bytes(b'{"a": 1}\n{"a": 2}')
        with self.assertRaises(SystemExit):
            K4.rows(p)
        p.write_bytes(b'{"a": 1}\n{"a": 2}\n')
        self.assertEqual(K4.rows(p), 2)

    def test_paths(self):
        self.refused(out_root=self.root / "student_khop3_c27", want="equals an input directory")
        self.refused(report=self.root / "eval/khop3", want="equals an input directory")
        self.refused(report=self.root / "eval/khop", want="equals an input directory")
        self.refused(report=self.d / "out/student_khop4_c27", want="equals an output directory")
        o = self.d / "out2/student_khop4_k1x"
        o.mkdir(parents=True)
        (o / "keep.txt").write_text("x")
        self.refused(out_root=self.d / "out2", want="holds other files")
        r = self.d / "rep2"
        r.mkdir()
        (r / "other.json").write_text("{}")
        self.refused(report=r, want="holds other files")


# --- the two stage-2 protections on the tiny model ---------------------------------------------------------------------
@unittest.skipUnless(HAVE_TORCH, "needs torch + transformers")
class TestStage2Protections(unittest.TestCase):
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
        kh, _, _, _, _ = B.build_train(random.Random(1), defaultdict(Counter), 8, 0)        # 24 khop rows
        assert len(gsm) == 6 and len(kh) == 24
        cls.gsm, cls.s1rows, cls.replay = gsm, kh[:18], kh[18:]
        w = lambda name, rows: (d / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        w("gsm.jsonl", gsm)
        w("kh.jsonl", cls.s1rows)
        w("replay.jsonl", cls.replay)
        w("dup.jsonl", cls.replay[:5] + [gsm[0]])                    # repeats a GSM training item_id
        w("ev.jsonl", gsm[:3] + kh[:5])
        (d / "tracer.py").write_text(K3T.TRACER)
        cls.s1 = cls.train_run(d / "s1", d / "m", d / "kh.jsonl", save=True)
        s1m = d / "s1/model"
        cls.ref = cls.train_run(d / "ref", s1m, d / "gsm.jsonl", lr="1e-3")
        cls.low = cls.train_run(d / "low", s1m, d / "gsm.jsonl", lr="2.5e-4")
        cls.rep = cls.train_run(d / "rep", s1m, d / "gsm.jsonl", lr="1e-3", extra=str(d / "replay.jsonl"))
        cls.dup = cls.train_run(d / "dup", s1m, d / "gsm.jsonl", lr="1e-3", extra=str(d / "dup.jsonl"), check=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def train_run(cls, out, model, train, lr="1e-3", extra="", save=False, check=True):
        trace = str(out) + ".trace.json"
        cmd = [*K3T.TORCHRUN, str(cls.d / "tracer.py"), TRAIN, "--model", str(model), "--train", str(train), "--eval",
               "val=" + str(cls.d / "ev.jsonl"), "--eval-subq", str(cls.d / "gsm.jsonl"), "--eval-value-render", "vt",
               "--final", "teacher", "--subq", "none", "--epochs", "1", "--micro-bs", "2", "--grad-accum", "2",
               "--max-len", "512", "--eval-max-len", "512", "--precision", "fp32master", "--lr", lr, "--log-every", "1",
               "--seed", "16", "--order-seed", "16", "--fsdp", "--fsdp-optim", "cpu-adamw", "--label-smoothing", "0.1",
               "--token-budget", "4096", "--out", str(out)]
        if extra:
            cmd += ["--extra-train", extra]
        if save:
            cmd += ["--save"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env={**K3T.ENV, "KHOP3_TRACE": trace})
        if check and r.returncode != 0:
            raise AssertionError(f"{out.name}: exit {r.returncode}\n{r.stderr[-3000:]}")
        res = {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "out": out}
        if r.returncode == 0:
            res["metrics"] = json.loads((out / "metrics.json").read_text())
            res["trace"] = json.loads(Path(trace).read_text())
        return res

    def test_stage1(self):
        self.assertIn("stage all: 18 examples per pass, 5 steps", self.s1["stdout"])
        self.assertEqual(self.s1["metrics"]["history"][-1]["step"], 5)
        self.assertTrue((self.d / "s1/model/config.json").exists())

    def test_l_quarter_lr(self):
        ref, low = self.ref["metrics"], self.low["metrics"]
        self.assertEqual((ref["args"]["lr"], low["args"]["lr"]), (1e-3, 2.5e-4))
        hr, hl = ref["history"], low["history"]
        self.assertEqual([x["step"] for x in hr], [1, 2])
        self.assertEqual([x["step"] for x in hl], [x["step"] for x in hr])
        self.assertEqual([x["lr"] for x in hl], [x["lr"] / 4 for x in hr])        # exact
        self.assertTrue(any(x["lr"] > 0 for x in hr))
        self.assertEqual(self.low["trace"], self.ref["trace"])                     # same examples per micro-batch
        self.assertGreater(len(self.ref["trace"]), 0)
        self.assertTrue(all(math.isfinite(x["loss"]) for x in hl))
        self.assertNotEqual([x["loss"] for x in hl], [x["loss"] for x in hr])     # the LR acts on the update
        self.assertIn("stage all: 6 examples per pass, 2 steps", self.low["stdout"])
        self.assertNotIn("extra-train:", self.low["stdout"])
        self.assertNotIn("extra_train", low)

    def test_r_replay(self):
        out, m = self.rep["stdout"], self.rep["metrics"]
        p = str(self.d / "replay.jsonl")
        self.assertRegex(out, rf"(?m)^extra-train: 6 rows from {p} ")
        self.assertIn(f"extra-train: total training rows 12 (6 from {self.d / 'gsm.jsonl'})", out)
        self.assertIn("stage all: 12 examples per pass, 3 steps", out)
        self.assertEqual(m["history"][-1]["step"], math.ceil((6 + 6) / 4))
        self.assertEqual(m["extra_train"], {p: 6})
        self.assertEqual(m["args"]["extra_train"], p)
        self.assertTrue(all(math.isfinite(x["loss"]) for x in m["history"]))
        ids = [e[0] for mb in self.rep["trace"] for e in mb]
        allowed = {r["item_id"] for r in self.gsm} | {r["item_id"] for r in self.replay}
        self.assertTrue(set(ids) <= allowed)
        self.assertEqual(set(ids), allowed)                                        # every row seen
        self.assertFalse(set(ids) & {r["item_id"] for r in self.s1rows})

    def test_duplicate_extra_train_stops(self):
        self.assertNotEqual(self.dup["rc"], 0)
        self.assertIn("is already a training row", self.dup["stdout"] + self.dup["stderr"])
        self.assertFalse((self.d / "dup/metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
