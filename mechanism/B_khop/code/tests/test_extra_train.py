"""--extra-train (CARD_khop_v1 section 1.4) in scripts/train_student.py and its smoke-stage subset in hpc/train_arm.sbatch.
CPU, the tiny random Qwen3.5 text model (8 layers, hidden 64) with the GPT-2 tokenizer (tests/test_depth_forcing.py).
Checks:
- flag off: history (loss per logged row), eval metrics and predictions byte-identical to the untouched snapshot
  (COTD_REF_REPO, default ../repo.base) and no "extra_train" key in metrics.json;
- flag on: the rows are appended right after the --train rows are read, before any shuffle: the run equals the
  reference trainer on one file holding the --train rows then the extra rows (same losses, same predictions); the
  log lines "extra-train: <n> rows from <path> (sources {...})" and "extra-train: total training rows <N>"; metrics.json
  extra_train {path: n}; two comma-separated files; a duplicate item_id (against --train or inside the extra rows) and a
  missing file are refused;
- one A2 step on mixed GSM + khop rows under 1-rank FSDP (torchrun --local-addr 127.0.0.1, cpu-adamw) with
  --eval-value-render vt and --eval-subq runs end to end;
- hpc/train_arm.sbatch with a stub python (records every call; no DRY_RUN xtrace, which macOS bash 3.2 cannot give):
  without --extra-train every call is identical to the snapshot's sbatch (smoke included); with it the smoke call trains
  on the first 16 rows of each extra file (one job-local file) and the arm call keeps the original path."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from test_depth_forcing import gpt2_tok, tiny_qwen35
    HAVE = True
except ImportError:
    HAVE = False
import build_khop  # noqa: E402

ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
TRAIN = str(ROOT / "scripts/train_student.py")
REF = Path(os.environ.get("COTD_REF_REPO") or ROOT.parent / "repo.base")
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
            "--local-addr", "127.0.0.1"]


def gsm_rows() -> list:
    return json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())["gsm8k"]


def khop_rows(n_per_task: int = 4) -> list:
    """KH-style training rows from the real generator (seeded), n_per_task per task."""
    import random
    from collections import Counter, defaultdict
    kh, _, _, _, _ = build_khop.build_train(random.Random(1), defaultdict(Counter), n_per_task, 0)
    return kh


def write(p, rows) -> str:
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return str(p)


def preds(out, name):
    return [json.loads(l) for l in open(Path(out) / f"preds_{name}.jsonl")]


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestExtraTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tok = gpt2_tok()
        except Exception as e:
            raise unittest.SkipTest(f"no local gpt2 tokenizer: {e}")
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = cls.tmp.name
        tiny_qwen35(len(cls.tok)).save_pretrained(d + "/m")
        cls.tok.save_pretrained(d + "/m")
        cls.gsm = write(d + "/gsm.jsonl", gsm_rows())
        cls.kh_rows = khop_rows()
        cls.kh = write(d + "/kh.jsonl", cls.kh_rows)
        cls.kh2 = write(d + "/kh2.jsonl", [dict(r, item_id=r["item_id"] + "-b") for r in cls.kh_rows[:5]])
        cls.both = write(d + "/both.jsonl", gsm_rows() + cls.kh_rows)
        cls.ev = write(d + "/ev.jsonl", gsm_rows()[:3] + cls.kh_rows[:5])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_train(self, out, *extra, train=None, script=TRAIN, ok=True):
        r = subprocess.run([sys.executable, script, "--model", self.d + "/m", "--train", train or self.gsm,
                            "--eval", "val=" + self.ev, "--final", "teacher", "--subq", "none", "--epochs", "2",
                            "--micro-bs", "2", "--grad-accum", "2", "--max-len", "512", "--eval-max-len", "512",
                            "--precision", "fp32master", "--lr", "1e-3", "--log-every", "1", "--seed", "1",
                            "--no-grad-ckpt", "--out", out, *extra], capture_output=True, text=True, timeout=900, env=ENV)
        if not ok:
            return r, None
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return r, json.loads(Path(out, "metrics.json").read_text())

    def ref_script(self):
        s = REF / "scripts/train_student.py"
        if not s.exists():
            self.skipTest(f"no reference checkout at {REF} (set COTD_REF_REPO)")
        return str(s)

    def test_flag_off_identical_to_reference(self):
        _, new = self.run_train(self.d + "/off_new")
        _, old = self.run_train(self.d + "/off_old", script=self.ref_script())
        self.assertEqual([r["loss"] for r in new["history"]], [r["loss"] for r in old["history"]])
        self.assertEqual(new["eval"], old["eval"])
        self.assertEqual(preds(self.d + "/off_new", "val"), preds(self.d + "/off_old", "val"))
        self.assertNotIn("extra_train", new)
        self.assertEqual(new["args"]["extra_train"], "")

    def test_appended_equals_concatenated_file(self):
        r, new = self.run_train(self.d + "/on", "--extra-train", self.kh)
        _, cat = self.run_train(self.d + "/cat", train=self.both, script=self.ref_script())
        self.assertIn(f"extra-train: {len(self.kh_rows)} rows from {self.kh} (sources {{'khop': {len(self.kh_rows)}}})",
                      r.stdout)
        self.assertIn(f"extra-train: total training rows {6 + len(self.kh_rows)} (6 from {self.gsm})", r.stdout)
        self.assertEqual(new["extra_train"], {self.kh: len(self.kh_rows)})
        self.assertEqual([x["loss"] for x in new["history"]], [x["loss"] for x in cat["history"]])
        self.assertEqual(new["eval"], cat["eval"])
        self.assertEqual(preds(self.d + "/on", "val"), preds(self.d + "/cat", "val"))

    def test_two_files(self):
        r, m = self.run_train(self.d + "/two", "--extra-train", f"{self.kh},{self.kh2}")
        self.assertEqual(m["extra_train"], {self.kh: len(self.kh_rows), self.kh2: 5})
        self.assertIn(f"total training rows {6 + len(self.kh_rows) + 5}", r.stdout)

    def test_refusals(self):
        dup = write(self.d + "/dup.jsonl", [gsm_rows()[0]])
        self_dup = write(self.d + "/selfdup.jsonl", self.kh_rows[:2] + self.kh_rows[:1])
        for arg, msg in ((dup, "is already a training row"), (self_dup, "is already a training row"),
                         (self.d + "/nofile.jsonl", "--extra-train: no file")):
            r, _ = self.run_train(self.d + "/bad", "--extra-train", arg, ok=False)
            self.assertNotEqual(r.returncode, 0, arg)
            self.assertIn(msg, r.stderr, arg)

    def test_fsdp_one_rank_mixed_rows_vt(self):
        out = self.d + "/fsdp"
        r = subprocess.run([*TORCHRUN, TRAIN, "--model", self.d + "/m", "--train", self.gsm, "--extra-train", self.kh,
                            "--eval", "val=" + self.ev, "--eval-subq", self.gsm, "--eval-value-render", "vt",
                            "--final", "teacher", "--subq", "none", "--epochs", "1", "--micro-bs", "2",
                            "--grad-accum", "2", "--max-len", "512", "--eval-max-len", "512", "--precision",
                            "fp32master", "--lr", "1e-3", "--log-every", "1", "--seed", "1", "--fsdp", "--fsdp-optim",
                            "cpu-adamw", "--label-smoothing", "0.1", "--out", out],
                           capture_output=True, text=True, timeout=900, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        m = json.loads(Path(out, "metrics.json").read_text())
        self.assertEqual(m["extra_train"], {self.kh: len(self.kh_rows)})
        self.assertEqual(m["fsdp"]["world"], 1)
        self.assertTrue(m["history"] and all(x["loss"] == x["loss"] for x in m["history"]))   # finite, not NaN
        self.assertEqual(m["eval"]["val"]["n"], 8)
        self.assertIn("subq", m["eval"])


STUB = r"""#!/bin/bash
# stub python for the sbatch test: record the call, keep a copy of every --extra-train file, pass the smoke check
n=$(( $(wc -l < "$HOME/calls.txt" 2>/dev/null || echo 0) + 1 ))
echo "$*" >> "$HOME/calls.txt"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [ "${args[i]}" = --extra-train ]; then cat ${args[i+1]//,/ } > "$HOME/extra_$n.jsonl"; fi
done
echo 'smoke {"n": 1}'
"""


def sbatch_run(home: Path, sbatch: Path, **env) -> list:
    """Run the sbatch with the stub python and the 663 env's keys (EXTRA_EVALS cut to one set); its python calls."""
    (home / "calls.txt").unlink(missing_ok=True)
    for f in home.glob("extra_*.jsonl"):
        f.unlink()
    base = {"NGPU": "1", "FSDP_OPTIM": "cpu-adamw", "TOKBUDGET": "4096", "ARMS": "A2", "DATA": "data/student_gsm8k_tf_vt",
            "EPOCHS": "1", "EVALAT": "0.5", "EVALMID": "val,valx", "EXTRA_EVALS": "valx=data/eval/valx.jsonl",
            "GA": "16", "LAMBDA": "1.0", "LR": "3e-6", "LSMOOTH": "0.1", "MBS": "2", "MODEL": "models/Qwen3.5-4B",
            "OPTIM": "adamw", "ORDERSEED": "16", "PREC": "bf16", "REPO": "repo", "SEED": "16", "TAG": "c1lr3ch",
            "XARGS": "--eval-value-render vt"}
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM", "_CONDOR", "SMOKE", "DRY_RUN", "XARGS"))}
    e.update(base)
    e.update(env)
    e.update({"HOME": str(home), "TMPDIR": str(home / "tmp")})
    r = subprocess.run(["bash", str(sbatch)], capture_output=True, text=True, timeout=120, env=e, cwd=str(home))
    if r.returncode != 0:
        raise AssertionError(r.stdout[-2000:] + r.stderr[-2000:])
    calls = (home / "calls.txt").read_text().splitlines()
    return [re.sub(r"tmp\.[A-Za-z0-9]+", "tmp.X", c) for c in calls]


class TestSbatchSmokeSubset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("bash"):
            raise unittest.SkipTest("no bash")
        cls.tmp = tempfile.TemporaryDirectory()
        h = cls.home = Path(cls.tmp.name)
        c = h / "cotd"
        (h / "tmp").mkdir()
        (c / "envs/cotd/bin").mkdir(parents=True)
        (c / "envs/cotd/bin/python").write_text(STUB)
        os.chmod(c / "envs/cotd/bin/python", 0o755)
        (c / "data/student_gsm8k_tf_vt").mkdir(parents=True)
        (c / "data/eval/khop").mkdir(parents=True)
        os.symlink(ROOT, c / "repo")
        lines = "".join(json.dumps(x) + "\n" for x in gsm_rows() * 4)
        for f in ("data/student_gsm8k_tf_vt/train.jsonl", "data/student_gsm8k_tf_vt/val.jsonl", "data/eval/valx.jsonl"):
            (c / f).write_text(lines)
        for f in ("kk_heldout", "jl_heldout", "jevbench_public", "typed_decisions_test", "bbeh"):
            (c / f"data/eval/{f}.jsonl").write_text(lines)
        kh = [dict(r, item_id=f"kh{i}") for i, r in enumerate(khop_rows(10))]   # 30 rows
        write(c / "data/eval/khop/train_kh.jsonl", kh)
        write(c / "data/eval/khop/train_k1.jsonl", [dict(r, item_id=r["item_id"] + "-k1") for r in kh])
        cls.kh = kh
        cls.sb = ROOT / "hpc/train_arm.sbatch"
        cls.ref = REF / "hpc/train_arm.sbatch"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_unchanged_without_extra_train(self):
        if not self.ref.exists():
            self.skipTest(f"no reference sbatch at {self.ref}")
        for env in ({}, {"SMOKE_ONLY": "1"}, {"XARGS": "--eval-value-render vt --extra-trainx 1"}):
            a = sbatch_run(self.home, self.ref, **env)
            b = sbatch_run(self.home, self.sb, **env)
            self.assertEqual(a, b, env)
            self.assertEqual(len(b), 1 if env.get("SMOKE_ONLY") else 2, b)

    def test_smoke_subset(self):
        calls = sbatch_run(self.home, self.sb, XARGS="--eval-value-render vt --extra-train data/eval/khop/train_kh.jsonl")
        self.assertEqual(len(calls), 2)
        smoke, arm = calls
        m = re.search(r"--extra-train (\S+/tmp\.X/extra_train\.jsonl)", smoke)    # the smoke stage's job-local dir
        self.assertIsNotNone(m, smoke)
        self.assertNotIn("data/eval/khop/train_kh.jsonl", smoke)
        self.assertIn("--extra-train data/eval/khop/train_kh.jsonl", arm)
        self.assertIn("--epochs 0.5", smoke)
        got = [json.loads(l)["item_id"] for l in open(self.home / "extra_1.jsonl")]
        self.assertEqual(got, [r["item_id"] for r in self.kh[:16]])
        # the only difference between the smoke call and the reference sbatch's smoke call is the extra-train path
        if self.ref.exists():
            ref_smoke = sbatch_run(self.home, self.ref,
                                   XARGS="--eval-value-render vt --extra-train data/eval/khop/train_kh.jsonl")[0]
            self.assertEqual(smoke.replace(m.group(1), "data/eval/khop/train_kh.jsonl"),
                             ref_smoke)

    def test_smoke_subset_two_files_and_smoke_only(self):
        calls = sbatch_run(self.home, self.sb, SMOKE_ONLY="1",
                           XARGS="--extra-train data/eval/khop/train_kh.jsonl,data/eval/khop/train_k1.jsonl "
                                 "--eval-value-render vt")
        self.assertEqual(len(calls), 1)
        self.assertRegex(calls[0], r"--extra-train \S+/tmp\.X/extra_train\.jsonl --eval-value-render vt")
        got = [json.loads(l)["item_id"] for l in open(self.home / "extra_1.jsonl")]
        self.assertEqual(got, [r["item_id"] for r in self.kh[:16]] + [r["item_id"] + "-k1" for r in self.kh[:16]])

    def test_missing_extra_file_stops_the_smoke(self):
        with self.assertRaises(AssertionError) as cm:
            sbatch_run(self.home, self.sb, XARGS="--extra-train data/eval/khop/none.jsonl")
        self.assertIn("missing --extra-train file data/eval/khop/none.jsonl", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
