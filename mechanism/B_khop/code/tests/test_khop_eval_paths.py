"""the k-hop design section 5: the two eval paths of the family give the same prompt ids and the
same letter probabilities on khop rows. Path 1: the in-job eval of scripts/train_student.py (EXTRA_EVALS: 1-rank FSDP
under torchrun, the FSDP2 batched readout, --eval-max-len at the trainer default 6144 because hpc/train_arm.sbatch passes
none). Path 2: scripts/eval_student.py on the saved weights as cloud run_eval_b2.sh calls it (--max-len 1536 --bs 8,
XARGS ''). CPU, the tiny random Qwen3.5 (tests/test_depth_forcing.py) with bf16-exact weights and --lr 0, so the saved
bf16 model holds exactly the weights the in-job eval used. Bar: max |log p1 - log p2| <= 1e-5 over 20 khop eval rows
(ca, wp, vb, cag; fwd and rev; blind and -qs rows included). Also: both scripts read the question through
student.final_text and student.evaluate, and no khop eval row is cut at 1536 tokens (the 6144 / 1536 difference is then
inert): checked with the GPT-2 tokenizer here and, when the Qwen3.5-4B tokenizer is in the HF cache, on every row of
the real khop eval files (E/src/data/eval/khop, KHOP_EVAL_DIR)."""
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    import transformers
    from cotdistill.student import encode_tail, final_text
    from test_depth_forcing import gpt2_tok, tiny_qwen35
    HAVE = True
except ImportError:
    HAVE = False
import build_khop  # noqa: E402

ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
            "--local-addr", "127.0.0.1"]
KHOP_EVAL_DIR = Path(os.environ.get("KHOP_EVAL_DIR") or ROOT.parents[1] / "src/data/eval/khop")
QWEN = "Qwen/Qwen3.5-4B"


def eval_rows(n: int = 20) -> list:
    """n khop eval rows from the real generator: every task, both orders, -qs and blind rows."""
    counts = {t: {1: 2, 3: 2} for t in ("ca", "wp", "vb")}
    counts["cag"] = {2: 2}
    sets, _ = build_khop.build_eval(random.Random(2), defaultdict(Counter), set(), set(), counts)
    rows = []
    for stem in ("khop_ca_fwd", "khop_ca_rev", "khop_wp_fwd", "khop_wp_rev", "khop_vb_fwd", "khop_vb_rev",
                 "khop_cag_fwd", "khop_ca_fwd_qs", "khop_blind"):
        rows += [dict(r, item_id=f"{stem}:{r['item_id']}") for r in sets[stem][:3]]
    return rows[:n]


@unittest.skipUnless(HAVE, "needs torch + transformers")
class TestEvalPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tok = gpt2_tok()
        except Exception as e:
            raise unittest.SkipTest(f"no local gpt2 tokenizer: {e}")

    def test_same_prompts_and_probs(self):
        rows = eval_rows()
        self.assertEqual(len(rows), 20)
        with tempfile.TemporaryDirectory() as d:
            m = tiny_qwen35(len(self.tok))
            with torch.no_grad():                    # bf16-exact weights: the bf16 save loses nothing
                for p in m.parameters():
                    p.copy_(p.to(torch.bfloat16).to(torch.float32))
            m.save_pretrained(d + "/m")
            self.tok.save_pretrained(d + "/m")
            ev = d + "/khop.jsonl"
            Path(ev).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            tr = d + "/train.jsonl"
            Path(tr).write_text("".join(json.dumps(r) + "\n" for r in
                                        json.loads((ROOT / "tests/fixtures/gsm8k_si_items_v1.json").read_text())["gsm8k"]))
            r = subprocess.run([*TORCHRUN, str(ROOT / "scripts/train_student.py"), "--model", d + "/m", "--train", tr,
                                "--eval", "khop=" + ev, "--final", "teacher", "--subq", "none", "--epochs", "1",
                                "--micro-bs", "2", "--grad-accum", "1", "--max-len", "1536", "--precision", "fp32master",
                                "--lr", "0", "--seed", "1", "--fsdp", "--fsdp-optim", "cpu-adamw", "--save",
                                "--out", d + "/job"], capture_output=True, text=True, timeout=900, env=ENV)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            args = json.loads(Path(d, "job/metrics.json").read_text())["args"]
            self.assertEqual(args["eval_max_len"], 6144)          # the sbatch passes no --eval-max-len
            self.assertEqual(args["eval_bs"], 8)
            r2 = subprocess.run([sys.executable, str(ROOT / "scripts/eval_student.py"), "--model", d + "/job/model",
                                 "--eval", "khop=" + ev, "--max-len", "1536", "--bs", "8", "--out", d + "/ev"],
                                capture_output=True, text=True, timeout=900, env=ENV)
            self.assertEqual(r2.returncode, 0, r2.stderr[-3000:])
            p1 = {x["item_id"]: x for x in map(json.loads, open(d + "/job/preds_khop.jsonl"))}
            p2 = {x["item_id"]: x for x in map(json.loads, open(d + "/ev/preds_khop.jsonl"))}
            self.assertEqual(sorted(p1), sorted(p2))
            self.assertEqual(len(p1), 20)
            worst = max(abs(math.log(a) - math.log(b)) for i in p1 for a, b in zip(p1[i]["probs"], p2[i]["probs"]))
            self.assertLessEqual(worst, 1e-5, worst)
            for i in p1:
                self.assertEqual((p1[i]["labels"], p1[i]["gold_label"]), (p2[i]["labels"], p2[i]["gold_label"]))
            for row in rows:                           # the same token ids at either max length (no cut)
                t = final_text(self.tok, row["prompt"])
                self.assertEqual(encode_tail(self.tok, t, 6144), encode_tail(self.tok, t, 1536))

    def test_both_scripts_use_the_same_readout(self):
        for s in ("train_student.py", "eval_student.py"):
            src = (ROOT / "scripts" / s).read_text()
            self.assertRegex(src, r"from cotdistill\.student import \([^)]*\bevaluate\b")
            self.assertRegex(src, r"from cotdistill\.student import \([^)]*\bfinal_text\b")

    def test_real_eval_files_fit_1536_qwen_tokens(self):
        files = sorted(KHOP_EVAL_DIR.glob("khop_*.jsonl"))
        if not files:
            self.skipTest(f"no khop eval files in {KHOP_EVAL_DIR}")
        try:
            tok = transformers.AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
        except Exception as e:
            self.skipTest(f"no local {QWEN} tokenizer: {e}")
        longest = 0
        for f in files:
            for l in open(f):
                t = final_text(tok, json.loads(l)["prompt"])
                ids = encode_tail(tok, t, 6144)
                longest = max(longest, len(ids))
                self.assertEqual(ids, encode_tail(tok, t, 1536), f.name)
        self.assertLess(longest, 1536)


if __name__ == "__main__":
    unittest.main()
