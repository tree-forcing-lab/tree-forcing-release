"""scripts/build_kk_subsample.py: fixed per-puzzle node subsets for the batch-28 node-density arms."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_kk_subsample as B  # noqa: E402


def items(n=40):
    out = []
    with open(ROOT / "data/student_kk_tf/train.jsonl") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) == n:
                break
    return out


class TestSubsample(unittest.TestCase):
    def test_subset_order_and_size(self):
        for it in items():
            for k in (3, 8):
                s = B.subsample(it, k)
                ids = [q["id"] for q in it["subqs"]]
                kept = [q["id"] for q in s["subqs"]]
                self.assertEqual(len(kept), min(k, len(ids)))
                self.assertEqual(kept, [i for i in ids if i in set(kept)])          # original order, no repeats
                self.assertEqual(s["subsample"], {"k": k, "n_full": len(ids)})
                self.assertEqual({x: s[x] for x in it if x != "subqs"}, {x: it[x] for x in it if x != "subqs"})

    def test_deterministic_and_item_keyed(self):
        it = items(1)[0]
        a, b = B.subsample(it, 3), B.subsample(dict(it), 3)
        self.assertEqual(a["subqs"], b["subqs"])
        other = dict(it, item_id=it["item_id"] + "x")
        self.assertNotEqual([q["id"] for q in B.subsample(other, 8)["subqs"]], [q["id"] for q in B.subsample(it, 8)["subqs"]])

    def test_small_tree_kept_whole(self):
        it = dict(items(1)[0]); it["subqs"] = it["subqs"][:2]
        self.assertEqual(B.subsample(it, 3)["subqs"], it["subqs"])

    def test_built_files(self):
        for k in (3, 8):
            d = ROOT / f"data/student_kk_tf_n{k}"
            if not (d / "summary.json").exists():
                self.skipTest(f"{d} not built")
            s = json.loads((d / "summary.json").read_text())
            self.assertEqual(s["md5"][f"data/student_kk_tf_n{k}/train.jsonl"], B.md5(d / "train.jsonl"))
            self.assertEqual(B.md5(d / "val.jsonl"), B.md5(ROOT / "data/student_kk_tf/val.jsonl"))
            self.assertEqual(s["train_after"]["nodes"], sum(min(k, len(it["subqs"])) for it in
                                                            map(json.loads, open(ROOT / "data/student_kk_tf/train.jsonl"))))


if __name__ == "__main__":
    unittest.main()
