"""Multiple-choice node views with the statements contexts (GFPS / GFPQ with --node-format mc, CHTC batch 19 jobs
1625-1627): an mc node view opens with the same context as the yes/no node views of its arm (the statement block, or
the problem without options), never with the full prompt; the --tree-view problem path and the yes/no statements path
are unchanged against 994ff50 (the commit before); the trainer runs GFPS + --node-format mc end to end. CPU only,
tiny random Qwen2 with the GPT-2 tokenizer; K&K items of data/student_kk_tf (skipped if absent)."""
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
BASE_REF = "994ff50"

try:
    import torch  # noqa: F401
    from cotdistill import student as st
    from cotdistill.treeplan import problem_no_options, statement_block
    from test_tree_forcing import ENV, SCRIPT, tiny
    HAVE = DATA.exists()
except ImportError:
    HAVE = False

GFPS = ["--tree-full", "plain", "--tree-view", "statements", "--tree-cap", "0", "--tree-sources", "knights_knaves"]
DUMP = r"""
import json, random, sys
sys.path.insert(0, sys.argv[1])
from cotdistill import student as st
items = [json.loads(l) for l in open(sys.argv[2])][:int(sys.argv[3])]
out = []
for view, fmt in (("problem", "mc"), ("statements", "yesno"), ("statements+question", "yesno"), ("problem", "yesno")):
    for it in items:
        rng = random.Random(7)
        ex = st.build_full_tree(it, mode="plain", rng=rng, cap=0, subq_target="cot", view=view, node_format=fmt,
                                sources={"knights_knaves"})
        out.append([view, fmt, [[e.text, list(e.labels), [round(x, 12) for x in e.target], round(e.weight, 12), e.kind]
                                for e in ex], rng.random()])
print(json.dumps(out))
"""


def items(n):
    with DATA.open() as f:
        return [json.loads(line) for line in f][:n]


@unittest.skipUnless(HAVE, "needs torch, transformers and data/student_kk_tf")
class TestMcStatements(unittest.TestCase):
    def test_context(self):
        for view, cut in (("statements", statement_block), ("statements+question", problem_no_options)):
            n_mc = n_nodes = 0
            for it in items(25):
                ex = st.build_full_tree(it, mode="plain", rng=random.Random(1), cap=0, subq_target="cot", view=view,
                                        node_format="mc", sources={"knights_knaves"})
                yn = st.build_full_tree(it, mode="plain", rng=random.Random(1), cap=0, subq_target="cot", view=view,
                                        node_format="yesno", sources={"knights_knaves"})
                nodes = [e for e in ex if e.kind == "subq"]
                self.assertEqual(len(nodes), len([e for e in yn if e.kind == "subq"]))   # one view per node
                self.assertEqual(ex[0].text, yn[0].text)                              # the final view unchanged
                ctx = cut(it)
                final_opts = it["prompt"].partition("\nOptions:\n")[2]
                for e in nodes:
                    self.assertTrue(e.text.startswith(ctx + "\n\n"), e.text[:200])
                    self.assertNotIn(final_opts, e.text)
                    self.assertAlmostEqual(sum(e.target), 1.0, places=6)
                    if "Intermediate question: Which of these questions has the answer Yes?" in e.text:
                        n_mc += 1
                        self.assertEqual(e.text.count("\nOptions:\n"), 1)
                n_nodes += len(nodes)
            self.assertGreater(n_mc, 0.8 * n_nodes, (view, n_mc, n_nodes))

    def test_other_paths_unchanged(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            subprocess.run(f"git -C {ROOT} archive {BASE_REF} src | tar x -C {d}", shell=True, check=True)
            runs = []
            for src in (d + "/src", str(ROOT / "src")):
                r = subprocess.run([sys.executable, "-c", DUMP, src, str(DATA), "12"], capture_output=True, text=True,
                                   env=ENV, timeout=600)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                runs.append(json.loads(r.stdout))
            self.assertEqual(runs[0], runs[1])

    def test_trainer_gfps_mc(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            tiny(d)
            Path(d + "/t.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items(2)))
            r = subprocess.run([sys.executable, SCRIPT, "--model", d + "/m", "--train", d + "/t.jsonl",
                                "--eval", "v=" + d + "/t.jsonl", "--epochs", "1", "--items-per-update", "2",
                                "--micro-bs", "2", "--max-len", "1024", "--eval-max-len", "1024",
                                "--precision", "fp32master", "--no-grad-ckpt", "--log-every", "1",
                                "--subq", "cot", "--subq-target", "cot", "--token-budget", "4096", "--out", d + "/o",
                                *GFPS, "--node-format", "mc"],
                               capture_output=True, text=True, timeout=1800, env=ENV)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            m = json.loads(Path(d + "/o/metrics.json").read_text())
            self.assertEqual(m["args"]["node_format"], "mc")
            self.assertEqual(m["args"]["tree_view"], "statements")
            self.assertGreater(m["tree_view_stats"]["views"], 0)


if __name__ == "__main__":
    unittest.main()
