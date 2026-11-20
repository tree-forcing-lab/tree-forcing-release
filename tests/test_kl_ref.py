"""--kl-ref (scripts/train_student.py, student.ref_kl): a KL(p || p_ref) term over each training example's options
against a frozen copy of the starting model. Off (the default) = byte-identical training to the commit before
(BASE_REF) on K&K items; on = every loss path runs (single GPU: full-tree, token-budget and fixed loops; FSDP2 over
gloo on the CPU), logs kl_ref, and a larger weight keeps the readout nearer the start. CPU only, tiny random Qwen2
with the GPT-2 tokenizer; K&K items of data/student_kk_tf (skipped if absent)."""
import json
import os
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
BASE_REF = "4e512a7"

try:
    import torch
    from cotdistill.student import Example, ref_kl
    import test_fsdp as tf_
    from test_fsdp import env, fsdp
    from test_tree_forcing import SCRIPT, tiny
    HAVE = DATA.exists()
except ImportError:
    HAVE = False

COMMON = ["--max-len", "1024", "--eval-max-len", "1024", "--log-every", "1", "--lr", "1e-3", "--seed", "3",
          "--label-smoothing", "0.1", "--epochs", "3"]
GFPS = ["--final", "teacher", "--subq", "cot", "--subq-target", "cot", "--tree-full", "plain", "--tree-view",
        "statements", "--tree-cap", "6", "--tree-sources", "knights_knaves", "--items-per-update", "2", "--micro-bs", "2"]
A2_BUDGET = ["--final", "teacher", "--subq", "none", "--micro-bs", "2", "--grad-accum", "2", "--token-budget", "4096"]
A2_FIXED = ["--final", "teacher", "--subq", "none", "--micro-bs", "2", "--grad-accum", "2"]


def stub_torchao(d: str) -> None:
    """The torchao stub of tests/test_fsdp.write_data (the FSDP path imports torchao.optim; CPU runs use the stub)."""
    (Path(d) / "stub/torchao/optim").mkdir(parents=True, exist_ok=True)
    (Path(d) / "stub/torchao/__init__.py").write_text('__version__ = "0.0.0"\n')
    (Path(d) / "stub/torchao/optim/__init__.py").write_text(tf_.tr.STUB)
    (Path(d) / "stub/torchao-0.0.0.dist-info").mkdir(parents=True, exist_ok=True)
    (Path(d) / "stub/torchao-0.0.0.dist-info/METADATA").write_text("Metadata-Version: 2.1\nName: torchao\nVersion: 0.0.0\n")


def items(n):
    with DATA.open() as f:
        return [json.loads(line) for line in f][:n]


def run(d, out, *argv, script=None):
    return subprocess.run([sys.executable, script or SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl", "--eval",
                           "v=" + d + "/t.jsonl", "--out", out, *COMMON, *argv],
                          capture_output=True, text=True, timeout=1800, env=env(d))


def hist(out):
    return json.loads((Path(out) / "metrics.json").read_text())["history"]


@unittest.skipUnless(HAVE, "needs torch, transformers and data/student_kk_tf")
class TestKlRef(unittest.TestCase):
    def test_ref_kl_math(self):
        z = [torch.tensor([1.0, 0.0, -1.0]), torch.tensor([0.5, 0.5])]
        ex = [Example("a", ["A", "B", "C"], [1, 0, 0], 1.0, "final", "x"), Example("b", ["A", "B"], [0, 1], 0.5, "subq", "x")]
        self.assertAlmostEqual(float(ref_kl(z, [t.clone() for t in z], ex)), 0.0, places=6)
        p = torch.softmax(z[0], -1)
        q = torch.softmax(torch.tensor([0.0, 0.0, 0.0]), -1)
        want = float((p * (p.log() - q.log())).sum()) / 2
        got = float(ref_kl(z, [torch.zeros(3), z[1].clone()], ex))
        self.assertAlmostEqual(got, want, places=5)

    def test_off_unchanged_and_on_runs(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            tiny(d)
            stub_torchao(d)
            Path(d + "/t.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items(4)))
            Path(d + "/base").mkdir()
            subprocess.run(f"git -C {ROOT} archive {BASE_REF} scripts src | tar x -C {d}/base", shell=True, check=True,
                           executable="/bin/bash")
            for name, args in (("gfps", GFPS), ("budget", A2_BUDGET), ("fixed", A2_FIXED)):
                r0 = run(d, f"{d}/{name}_base", *args, script=d + "/base/scripts/train_student.py")
                r1 = run(d, f"{d}/{name}_off", *args)
                self.assertEqual(r0.returncode, 0, r0.stderr[-1500:])
                self.assertEqual(r1.returncode, 0, r1.stderr[-1500:])
                h0, h1 = hist(f"{d}/{name}_base"), hist(f"{d}/{name}_off")
                self.assertEqual([x["loss"] for x in h0], [x["loss"] for x in h1], name)   # off: same numbers
                self.assertTrue(all("kl_ref" not in x for x in h1))
                kls = {}
                for beta in ("0.01", "100"):
                    r = run(d, f"{d}/{name}_kl{beta}", *args, "--kl-ref", beta)
                    self.assertEqual(r.returncode, 0, r.stderr[-1500:])
                    self.assertIn("frozen reference", r.stdout)
                    h = hist(f"{d}/{name}_kl{beta}")
                    self.assertTrue(all("kl_ref" in x for x in h))
                    kls[beta] = h[-1]["kl_ref"]
                    self.assertAlmostEqual(h[0]["kl_ref"], 0.0, places=6)    # the first update: model = reference
                self.assertGreater(kls["0.01"], 0.0, name)
                self.assertLess(kls["100"], kls["0.01"], name)               # a larger weight stays nearer the start

    def test_fsdp_path(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            tiny(d)
            stub_torchao(d)
            Path(d + "/t.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items(4)))
            base = ["--model", d + "/m", "--train", d + "/t.jsonl", "--eval", "v=" + d + "/t.jsonl", *COMMON, "--precision",
                    "fp32master", "--fsdp-optim", "cpu-adamw"]
            for name, args in (("gfps", GFPS), ("a2", A2_BUDGET)):
                off = fsdp(d, *base, *args, "--out", f"{d}/f_{name}_off", nproc=1)
                on = fsdp(d, *base, *args, "--out", f"{d}/f_{name}_on", "--kl-ref", "1.0", nproc=1)
                self.assertEqual(off.returncode, 0, off.stderr[-1500:])
                self.assertEqual(on.returncode, 0, on.stderr[-1500:])
                h0, h1 = hist(f"{d}/f_{name}_off"), hist(f"{d}/f_{name}_on")
                self.assertTrue(all("kl_ref" in x for x in h1) and all("kl_ref" not in x for x in h0))
                self.assertEqual(h1[0]["kl_ref"], 0.0)                      # the first update: model = reference
                self.assertGreater(max(x["kl_ref"] for x in h1), 0.0, (name, [x["kl_ref"] for x in h1]))

    def test_refused_combinations(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            tiny(d)
            Path(d + "/t.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items(2)))
            r = run(d, f"{d}/x", *GFPS, "--kl-ref", "-1")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--kl-ref", r.stderr)


if __name__ == "__main__":
    unittest.main()
