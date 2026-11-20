"""--rationale-lm with tree nodes and --gen-eval (scripts/train_student.py, student.rationale_example,
student.generate_rationale, student.parse_generated): the LM example on the reasoning and answer joins each item's
full-tree group (one per item with a rationale), the full-tree loop scores it apart with lm_loss; without the flag
every path is byte-identical to BASE_REF (full-tree GFPS and the grouped A7 arm); --gen-eval writes greedy
reasoning-then-answer predictions. CPU only, tiny random Qwen2 with the GPT-2 tokenizer; K&K items of
data/student_kk_tf with a stand-in rationale (skipped if absent)."""
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
DATA = Path(os.environ.get("KK_TF_DATA", ROOT / "data/student_kk_tf/train.jsonl"))
BASE_REF = "b922edf"

try:
    import torch  # noqa: F401
    from cotdistill.student import build_full_tree, parse_generated
    from test_fsdp import env, fsdp
    from test_kl_ref import stub_torchao
    from test_tree_forcing import SCRIPT, tiny
    HAVE = DATA.exists()
except ImportError:
    HAVE = False

COMMON = ["--max-len", "1024", "--eval-max-len", "1024", "--log-every", "1", "--lr", "1e-3", "--seed", "3",
          "--label-smoothing", "0.1", "--epochs", "1", "--lm-max-len", "1536"]
GFPS = ["--final", "teacher", "--subq", "cot", "--subq-target", "cot", "--tree-full", "plain", "--tree-view",
        "statements", "--tree-cap", "4", "--tree-sources", "knights_knaves", "--items-per-update", "2", "--micro-bs", "2"]
A7 = ["--final", "teacher", "--subq", "none", "--micro-bs", "2", "--grad-accum", "2", "--rationale-lm"]


def items(n, rationale=True):
    with DATA.open() as f:
        out = [json.loads(line) for line in f][:n]
    for i, it in enumerate(out):
        if rationale:
            it["rationale"] = f"Step {i}: check each statement against each option; only one option is consistent."
    return out


def run(d, out, *argv, script=None):
    return subprocess.run([sys.executable, script or SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl", "--eval",
                           "v=" + d + "/t.jsonl", "--out", out, *COMMON, *argv],
                          capture_output=True, text=True, timeout=1800, env=env(d))


def hist(out):
    return json.loads((Path(out) / "metrics.json").read_text())["history"]


class TestParse(unittest.TestCase):
    def test_parse_generated(self):
        L = ["A", "B", "C"]
        self.assertEqual(parse_generated(" 3 + 4 = 7.\n\nAnswer: B", L), "B")
        self.assertEqual(parse_generated("so Answer: (c) and later Answer: A", L), "C")
        self.assertIsNone(parse_generated("Answer: D", L))
        self.assertIsNone(parse_generated("Answer: Because", L))
        self.assertIsNone(parse_generated("no answer here", L))


@unittest.skipUnless(HAVE, "needs torch, transformers and data/student_kk_tf")
class TestCotTree(unittest.TestCase):
    def test_group_has_one_lm_example(self):
        for it in items(3):
            g0 = build_full_tree(it, mode="plain", rng=random.Random(1), cap=4, view="statements",
                                 sources={"knights_knaves"})
            g1 = build_full_tree(it, mode="plain", rng=random.Random(1), cap=4, view="statements",
                                 sources={"knights_knaves"}, rationale_lm=True, lambda_lm=0.5)
            self.assertEqual([e.text for e in g1[:-1]], [e.text for e in g0])     # the views unchanged
            lm = g1[-1]
            self.assertEqual(lm.kind, "lm")
            self.assertEqual(lm.weight, 0.5)
            self.assertTrue(lm.text.endswith("\n\nReasoning:") and lm.text.startswith(it["prompt"]))
            gold = max(it["teacher"], key=it["teacher"].get)
            self.assertTrue(lm.continuation.endswith(f"\n\nAnswer: {gold}"))
        it = items(1, rationale=False)[0]
        g = build_full_tree(it, mode="plain", rng=random.Random(1), cap=4, view="statements",
                            sources={"knights_knaves"}, rationale_lm=True)
        self.assertFalse(any(e.kind == "lm" for e in g))                             # no rationale: no LM example

    def test_off_unchanged_on_runs_and_gen_eval(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            tiny(d)
            Path(d + "/t.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items(4)))
            Path(d + "/base").mkdir()
            subprocess.run(f"git -C {ROOT} archive {BASE_REF} scripts src | tar x -C {d}/base", shell=True,
                           check=True, executable="/bin/bash")
            for name, args in (("gfps", GFPS), ("a7", A7)):                 # off (and the A7 refactor): unchanged
                r0 = run(d, f"{d}/{name}_base", *args, script=d + "/base/scripts/train_student.py")
                r1 = run(d, f"{d}/{name}_new", *args)
                self.assertEqual(r0.returncode, 0, r0.stderr[-1500:])
                self.assertEqual(r1.returncode, 0, r1.stderr[-1500:])
                self.assertEqual([x["loss"] for x in hist(f"{d}/{name}_base")],
                                 [x["loss"] for x in hist(f"{d}/{name}_new")], name)
            r = run(d, f"{d}/cot", *GFPS, "--rationale-lm", "--gen-eval", f"g={d}/t.jsonl", "--gen-max-new", "6",
                    "--gen-bs", "3")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            h = hist(f"{d}/cot")
            self.assertTrue(all(x.get("loss_lm", 0) > 0 for x in h))
            h0 = hist(f"{d}/gfps_new")
            self.assertTrue(all(x["loss"] > y["loss"] for x, y in zip(h, h0)) or len(h) != len(h0))
            m = json.loads((Path(d) / "cot/metrics.json").read_text())
            g = m["gen_eval"]["g"]
            self.assertEqual(g["n"], 4)
            self.assertLessEqual(g["mean_new_tokens"], 6)
            rows = [json.loads(x) for x in open(Path(d) / "cot/preds_gen_g.jsonl")]
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertAlmostEqual(sum(row["probs"]), 1.0, places=6)
            bad = run(d, f"{d}/bad", *GFPS, "--rationale-lm", "--tree-pack", "4")
            self.assertNotEqual(bad.returncode, 0)


@unittest.skipUnless(HAVE, "needs torch, transformers and data/student_kk_tf")
class TestCotTreeFsdp(unittest.TestCase):
    """The same through --fsdp (FSDP2 over gloo on the CPU, one rank): the LM examples go through the root's run(),
    --gen-eval gathers the full model into a plain copy."""

    def test_fsdp(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            tiny(d)
            stub_torchao(d)
            Path(d + "/t.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items(4)))
            Path(d + "/base").mkdir()
            subprocess.run(f"git -C {ROOT} archive {BASE_REF} scripts src | tar x -C {d}/base", shell=True,
                           check=True, executable="/bin/bash")
            base = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl", *COMMON,
                    "--precision", "fp32master", "--fsdp-optim", "cpu-adamw"]
            off0 = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
                                   d + "/base/scripts/train_student.py", *base, *GFPS, "--out", f"{d}/f0", "--fsdp"],
                                  capture_output=True, text=True, timeout=1800, env=env(d))
            off1 = fsdp(d, *base, *GFPS, "--out", f"{d}/f1", nproc=1)
            self.assertEqual(off0.returncode, 0, off0.stderr[-1500:])
            self.assertEqual(off1.returncode, 0, off1.stderr[-1500:])
            self.assertEqual([x["loss"] for x in hist(f"{d}/f0")], [x["loss"] for x in hist(f"{d}/f1")])
            on = fsdp(d, *base, *GFPS, "--rationale-lm", "--gen-eval", f"g={d}/t.jsonl", "--gen-max-new", "6",
                      "--gen-bs", "3", "--out", f"{d}/f2", nproc=1)
            self.assertEqual(on.returncode, 0, on.stderr[-2500:])
            h = hist(f"{d}/f2")
            self.assertTrue(all(x.get("loss_lm", 0) > 0 for x in h))
            single = run(d, f"{d}/s2", *GFPS, "--rationale-lm", "--precision", "fp32master")
            self.assertEqual(single.returncode, 0, single.stderr[-1500:])
            hs = hist(f"{d}/s2")
            self.assertAlmostEqual(h[0]["loss"], hs[0]["loss"], places=4)      # update 1: same weights, same loss
            self.assertAlmostEqual(h[0]["loss_lm"], hs[0]["loss_lm"], places=4)
            m = json.loads((Path(d) / "f2/metrics.json").read_text())
            self.assertEqual(m["gen_eval"]["g"]["n"], 4)
            self.assertTrue((Path(d) / "f2/preds_gen_g.jsonl").exists())
            a7 = fsdp(d, *base, *A7, "--out", f"{d}/f3", nproc=1)
            self.assertEqual(a7.returncode, 0, a7.stderr[-1500:])
            two = fsdp(d, *base, *GFPS, "--rationale-lm", "--out", f"{d}/f4", nproc=2)
            self.assertNotEqual(two.returncode, 0)
            self.assertIn("one rank only", two.stderr + two.stdout)


if __name__ == "__main__":
    unittest.main()
