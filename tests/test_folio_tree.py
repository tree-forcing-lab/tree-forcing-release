"""FOLIO with prover-verified atom nodes as a Tree Forcing source (treeplan.FINAL_QUESTION["folio_tf"], the cluster batch
27): the statement cut takes the preamble and the "- <premise>" lines and drops the final question paragraph (which
names the conclusion) and the options; the GFPS views are that block plus one intermediate question per node; the
old "folio" cut and the K&K cut are unchanged. CPU only, standard library + the package."""
import json
import os
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("FOLIO_TF_DATA", ROOT / "data/student_folio_tf/train.jsonl"))
TEST = Path(os.environ.get("FOLIO_TF_TEST", ROOT / "data/eval/folio_tf_test.jsonl"))
KK_DATA = ROOT / "data/student_kk_tf/train.jsonl"

FINAL = "Question: Based on the statements, is the following conclusion true, false, or uncertain? Conclusion: "
BLOCK = ("Each statement below is true. Judge the conclusion using only these statements and valid logical reasoning: "
         "it is true if the statements imply it, false if they imply its negation, and uncertain otherwise.\n\n"
         "- All people who regularly drink coffee are dependent on caffeine.\n- Rina is a student.")
NODES = [{"id": "n1", "question": 'Is it true that Rina has the property "student"?',
          "opposite": 'Is it false that Rina has the property "student"?', "truth": True, "p_cot": 1.0,
          "type": "parse", "depends_on": [], "node_depth": 0},
         {"id": "n2", "question": 'Is it false that Rina stands in the relation "is dependent on" to caffeine?',
          "opposite": 'Is it true that Rina stands in the relation "is dependent on" to caffeine?', "truth": False,
          "p_cot": 0.0, "type": "derive", "depends_on": [], "node_depth": 0}]


def item(conclusion="Rina is dependent on caffeine.", final=FINAL):
    prompt = BLOCK + "\n\n" + final + conclusion + "\nOptions:\nA) True\nB) False\nC) Uncertain"
    return {"item_id": "ftf-test", "source": "folio_tf", "prompt": prompt, "labels": ["A", "B", "C"],
            "gold_label": "C", "teacher": {"A": 0.0, "B": 0.0, "C": 1.0}, "subqs": [dict(x) for x in NODES]}


class TestFolioCut(unittest.TestCase):
    def test_pattern(self):
        from cotdistill.treeplan import FINAL_QUESTION
        pat = FINAL_QUESTION["folio_tf"]
        self.assertTrue(re.fullmatch(pat, FINAL + "Rina is not a student."))
        for bad in (FINAL, FINAL + "a\nb", "Question: Is the statement true, false, or uncertain given the premises?",
                    FINAL.replace("Based on the statements, ", "") + "Rina is a student."):
            self.assertIsNone(re.fullmatch(pat, bad))

    def test_statement_block(self):
        from cotdistill.treeplan import problem_no_options, statement_block
        it = item()
        self.assertEqual(statement_block(it), BLOCK)
        self.assertNotIn("Conclusion:", statement_block(it))
        self.assertEqual(problem_no_options(it), BLOCK + "\n\n" + FINAL + "Rina is dependent on caffeine.")
        with self.assertRaises(ValueError):
            statement_block(item(final="Question: Is it true? Conclusion: "))

    def test_old_folio_cut_unchanged(self):
        from cotdistill.treeplan import statement_block
        body = "Read the premises.\n\nPremises:\nA.\n\nStatement: B."
        old = {"item_id": "f", "source": "folio", "prompt": body + "\n\nQuestion: Is the statement true, false, or "
               "uncertain given the premises?\nOptions:\nA) True\nB) False\nC) Uncertain"}
        self.assertEqual(statement_block(old), body)

    def test_gfps_views(self):
        from cotdistill.student import build_full_tree
        it = item()
        ex = build_full_tree(it, mode="plain", view="statements", rng=random.Random(0), cap=0, sources={"folio_tf"})
        self.assertEqual(ex[0].kind, "final")
        self.assertEqual(ex[0].text, it["prompt"] + "\n\nAnswer:")
        self.assertEqual(len(ex), 1 + len(NODES))
        for e in ex[1:]:
            self.assertTrue(e.text.startswith(BLOCK + "\n\nIntermediate question: "), e.text[:80])
            self.assertNotIn("Conclusion:", e.text)
            self.assertNotIn("C) Uncertain", e.text)

    @unittest.skipUnless(DATA.exists(), "needs data/student_folio_tf (scripts/build_folio_tf.py on main)")
    def test_built_items(self):
        from cotdistill.student import build_full_tree
        from cotdistill.treeplan import statement_block
        for path in (DATA, TEST):
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                it = json.loads(line)
                block = statement_block(it)
                self.assertTrue(it["prompt"].startswith(block + "\n\n" + FINAL))
                self.assertNotIn("Conclusion:", block)        # (a conclusion may restate a premise: FOLIO True items)
                ex = build_full_tree(it, mode="plain", view="statements", rng=random.Random(1), cap=0,
                                     sources={"folio_tf"})
                self.assertEqual(len(ex), 1 + len(it["subqs"]))
                self.assertTrue(all(e.text.startswith(block + "\n\nIntermediate question: ") for e in ex[1:]))

    @unittest.skipUnless(KK_DATA.exists(), "needs data/student_kk_tf")
    def test_kk_cut_unchanged(self):
        from cotdistill.treeplan import statement_block
        for line in KK_DATA.read_text().splitlines()[:50]:
            it = json.loads(line)
            self.assertEqual(statement_block(it), it["prompt"].split("\n\nQuestion: ")[0])


if __name__ == "__main__":
    unittest.main()
