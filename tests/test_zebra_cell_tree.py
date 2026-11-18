"""Zebra cell questions as a Tree Forcing source (treeplan.FINAL_QUESTION["zebra_tf"], CHTC batch 24): the statement
cut accepts the one-cell question of zebra.make_cell_item (main's scripts/build_zebra_cell_tf.py) and still the
full-arrangement question; the GFPS views of a cell item are the puzzle (without the question and its options) plus
one intermediate question per tree node; other sources are cut as before. CPU only, standard library + the package."""
import json
import os
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("ZEBRA_CELL_DATA", ROOT / "data/student_zebra_cell_tf/train.jsonl"))
KK_DATA = ROOT / "data/student_kk_tf/train.jsonl"

LABELS = ["name", "house color", "pet", "drink", "sport", "job", "instrument", "favorite fruit"]   # zebra.LABEL
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth"]                                # zebra.ORDINAL
PUZZLE = ("There are 3 houses in a row, numbered 1 to 3 from left to right. Each house is home to exactly one person, "
          "and no two people share any of the following attributes.\n- Each person has a different first name: Greta, "
          "Lena, Victor\n- Each person plays a different sport: golf, hockey, soccer\n\nClues:\n1. The person who plays "
          "hockey is Victor.\n2. Greta lives directly to the left of the person who plays soccer.")
NODES = [{"id": "n1", "question": "According to clue 2, could Greta live in the first house if the person who plays soccer "
                                  "lived in the second house?", "opposite": None, "truth": True, "p_cot": 1.0,
          "type": "parse", "depends_on": []},
         {"id": "n2", "question": "Does Greta live in the first house?", "opposite": "Does Greta live somewhere other "
          "than the first house?", "truth": True, "p_cot": 1.0, "type": "derive", "depends_on": ["n1"]}]


def cell_item(question="Question: What is the sport of the person who lives in the first house?"):
    prompt = PUZZLE + "\n\n" + question + "\nOptions:\nA) hockey\nB) golf\nC) soccer"
    return {"item_id": "zc-test", "source": "zebra_tf", "prompt": prompt, "labels": ["A", "B", "C"], "gold_label": "B",
            "teacher": {"A": 0.0, "B": 1.0, "C": 0.0}, "subqs": [dict(x) for x in NODES]}


class TestZebraCellCut(unittest.TestCase):
    def test_every_cell_question_matches(self):
        from cotdistill.treeplan import FINAL_QUESTION
        pat = FINAL_QUESTION["zebra_tf"]
        for lab in LABELS:
            for o in ORDINALS:
                self.assertTrue(re.fullmatch(pat, f"Question: What is the {lab} of the person who lives in the {o} house?"))
        self.assertTrue(re.fullmatch(pat, "Question: Which option gives the correct arrangement?"))
        for bad in ("Question: What is the sport of the person who lives in the seventh house?",
                    "Question: What is the color of the person who lives in the first house?",
                    "Question: Which option gives the correct arrangement",
                    "Question: Who lives in the first house?"):
            self.assertIsNone(re.fullmatch(pat, bad))

    def test_statement_block(self):
        from cotdistill.treeplan import statement_block
        self.assertEqual(statement_block(cell_item()), PUZZLE)
        full = dict(cell_item(), prompt=PUZZLE + "\n\nQuestion: Which option gives the correct arrangement?\nOptions:\n"
                    "A) House 1: Greta, golf; House 2: ...\nB) ...")
        self.assertEqual(statement_block(full), PUZZLE)
        with self.assertRaises(ValueError):
            statement_block(cell_item("Question: Who plays golf?"))

    def test_gfps_views(self):
        from cotdistill.student import build_full_tree
        it = cell_item()
        ex = build_full_tree(it, mode="plain", view="statements", rng=random.Random(0), cap=0, sources={"zebra_tf"})
        self.assertEqual(ex[0].kind, "final")
        self.assertEqual(ex[0].text, it["prompt"] + "\n\nAnswer:")
        self.assertEqual(len(ex), 1 + len(NODES))
        for e in ex[1:]:
            self.assertTrue(e.text.startswith(PUZZLE + "\n\nIntermediate question: "), e.text[:80])
            self.assertNotIn("Question: What is the sport", e.text)
            self.assertNotIn("A) hockey", e.text)

    @unittest.skipUnless(DATA.exists(), "needs data/student_zebra_cell_tf (scripts/build_zebra_cell_tf.py on main)")
    def test_built_items(self):
        from cotdistill.student import build_full_tree
        from cotdistill.treeplan import statement_block
        rows = [json.loads(line) for line in DATA.read_text().splitlines()[:400] if line.strip()]
        for it in rows:
            block = statement_block(it)
            self.assertTrue(it["prompt"].startswith(block + "\n\nQuestion: What is the "))
            ex = build_full_tree(it, mode="plain", view="statements", rng=random.Random(1), cap=0, sources={"zebra_tf"})
            self.assertEqual(len(ex), 1 + len([x for x in it["subqs"] if "id" in x]))
            self.assertTrue(all(e.text.startswith(block + "\n\nIntermediate question: ") for e in ex[1:]))

    @unittest.skipUnless(KK_DATA.exists(), "needs data/student_kk_tf")
    def test_kk_cut_unchanged(self):
        from cotdistill.treeplan import statement_block
        for line in KK_DATA.read_text().splitlines()[:50]:
            it = json.loads(line)
            self.assertEqual(statement_block(it), it["prompt"].split("\n\nQuestion: ")[0])


if __name__ == "__main__":
    unittest.main()
