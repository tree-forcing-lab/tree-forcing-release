"""scripts/build_folio_tf.py: the FOL parser, the prover labels, the verified atom nodes, the item
format, the story-grouped splits and a byte-identical rebuild. Needs z3-solver (~/.venvs/sev).
 ~/.venvs/sev/bin/python -m unittest tests/test_build_folio_tf.py"""
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_folio_tf as B  # noqa: E402
import z3  # noqa: E402

DATA = ROOT / "data/student_folio_tf"
HAVE = (DATA / "train.jsonl").exists() and (ROOT / "data/raw/folio/folio_v2_train.jsonl").exists()
# treeplan.FINAL_QUESTION["folio_tf"] on branch folio (tests/test_folio_tree.py there)
FINAL_RE = r"Question: Based on the statements, is the following conclusion true, false, or uncertain\? Conclusion: [^\n]+"


def row(prem, concl, label="True", premises_nl="Rina is a student.", conclusion_nl="Rina is a student."):
    return {"story_id": 1, "example_id": 1, "premises": premises_nl, "conclusion": conclusion_nl,
            "premises-FOL": "\n".join(prem), "conclusion-FOL": concl, "label": label}


def load(p):
    return [json.loads(line) for line in open(p)]


class TestParser(unittest.TestCase):
    def test_connectives_and_scope(self):
        f = B.parse("∀x (A(x) → B(x) ∨ C(x))")
        self.assertEqual(f[0], "forall")
        self.assertEqual(f[2][0], "imp")
        self.assertEqual(f[2][2][0], "or")
        self.assertEqual(B.parse("A(a) ⊕ B(a)")[0], "xor")
        self.assertEqual(B.parse("A(a) ↔ ¬B(a)")[0], "iff")
        self.assertEqual(B.parse("A(a) → B(a) → C(a)")[2][0], "imp")          # right associative
        self.assertEqual(B.parse("∃x∃y (R(x, y) ∧ ¬(x=y))")[2][0], "exists")
        self.assertEqual(B.parse("∀x ∀y (x≠y → R(x, y))")[2][2][1][0], "not")     # ≠ is ¬(=)
        self.assertEqual(B.parse("A(a) ∧ B(a) ∨ C(a)")[0], "or")             # ∧ binds tighter than ∨
        self.assertEqual(B.parse("Building(świątek)")[2][0], ("const", "świątek"))

    def test_quantifier_scope(self):
        f = B.parse("∀x (∃y (R(x, y)) → A(x))")                         # ∃ binds its parenthesised group only
        self.assertEqual(f[2][0], "imp")
        self.assertEqual(f[2][1][0], "exists")
        self.assertEqual(B.parse("∃x∃y (R(x, y)) ∧ A(c)")[0], "and")
        fx = []
        f = B.parse("∀x (A(x)) → B(x)", fx)                              # narrow leaves x free: the maximal reading
        self.assertEqual((f[0], fx), ("forall", ["scope_maximal"]))
        self.assertEqual(B.free_vars(f), set())
        self.assertEqual(B.parse("∀x ¬A(x) ∨ B(x)")[0], "forall")         # no parenthesis: maximal

    def test_repairs_counted(self):
        fx = []
        self.assertEqual(B.parse("A(a) ∧ (B(a) ∨ C(a)", fx)[0], "and")
        self.assertEqual(fx, ["missing_close"])
        fx = []
        B.parse("A(a) ∧ B(a))", fx)
        self.assertEqual(fx, ["surplus_close"])
        fx = []
        B.parse("R(a, b,)", fx)
        self.assertEqual(fx, ["trailing_comma"])
        with self.assertRaises(B.ParseError):
            B.parse("A(a) ∧ ∧ B(a)")

    def test_free_variable(self):
        self.assertEqual(B.free_vars(B.parse("MeansTo(x, drive) ∨ Student(james)")), {"x"})
        self.assertEqual(B.analyse(row(["MeansTo(x, drive) ∨ Student(james)"], "Student(james)"))["drop"],
                         "free_variable")


class TestProver(unittest.TestCase):
    def test_labels(self):
        prem = ["∀x (Student(x) → Young(x))", "Student(rina)"]
        self.assertEqual(B.analyse(row(prem, "Young(rina)"))["label"], "True")
        self.assertEqual(B.analyse(row(prem, "¬Young(rina)"))["label"], "False")
        self.assertEqual(B.analyse(row(prem, "Tall(rina)"))["label"], "Uncertain")
        self.assertEqual(B.analyse(row(["A(ann)", "¬A(ann)"], "B(ann)"))["drop"], "premises_unsat")

    def test_nodes_and_leaks(self):
        prem = ["∀x (Student(x) → Young(x))", "Student(rina) ∧ Tall(rina)", "∀x (Young(x) → Happy(x))"]
        r = B.analyse(row(prem, "Happy(rina)"))
        atoms = {(n, a): (t, k) for n, a, t, k in r["nodes"]}
        self.assertNotIn(("Happy", ("rina",)), atoms)                          # the conclusion itself: a leak
        self.assertEqual(r["leak_dropped"], 1)
        self.assertEqual(atoms[("Student", ("rina",))], (True, "parse"))       # a conjunct of a premise
        self.assertEqual(atoms[("Young", ("rina",))], (True, "derive"))
        r = B.analyse(row(prem, "¬¬Happy(rina) ∧ Happy(rina)"))
        self.assertNotIn(("Happy", ("rina",)), {(n, a) for n, a, _, _ in r["nodes"]})   # equivalent, not equal

    def test_node_text(self):
        self.assertEqual(B.claim("IsDependentOn", ["Rina", "caffeine"]),
                         'Rina stands in the relation "is dependent on" to caffeine')
        self.assertEqual(B.claim("ChoralConductor", ["Miroslav"]), 'Miroslav has the property "choral conductor"')
        self.assertEqual(B.const_text("miroslavVenhoda", "Miroslav Venhoda was a Czech choral conductor."),
                         "Miroslav Venhoda")


@unittest.skipUnless(HAVE, "needs data/student_folio_tf (python scripts/build_folio_tf.py) and data/raw/folio")
class TestBuilt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = {"train": load(DATA / "train.jsonl"), "val": load(DATA / "val.jsonl"),
                     "test": load(ROOT / "data/eval/folio_tf_test.jsonl"),
                     "test_all": load(ROOT / "data/eval/folio_tf_test_all.jsonl")}
        cls.raw = {}
        for f in ("folio_v2_train.jsonl", "folio_v2_validation.jsonl"):
            for r in load(ROOT / "data/raw/folio" / f):
                cls.raw[(f, r["example_id"])] = r

    def test_format(self):
        for name, xs in self.split.items():
            for x in xs:
                head, sep, opts = x["prompt"].partition("\nOptions:\n")
                self.assertEqual(opts, "A) True\nB) False\nC) Uncertain")
                body, _, last = head.rpartition("\n\n")
                self.assertTrue(re.fullmatch(FINAL_RE, last), last)
                pre, _, lines = body.partition("\n\n")
                self.assertEqual(pre, B.PREAMBLE)
                self.assertTrue(all(ln.startswith("- ") for ln in lines.split("\n")))
                self.assertEqual(x["source"], "folio_tf")
                self.assertEqual(x["teacher"], {L: float(L == x["gold_label"]) for L in "ABC"})
                self.assertEqual(x["label_names"][x["gold_label"]], x["group"].split("/")[1])

    def test_gold_and_prover(self):
        f = {"train": "folio_v2_train.jsonl", "val": "folio_v2_train.jsonl", "test": "folio_v2_validation.jsonl",
             "test_all": "folio_v2_validation.jsonl"}
        for name, xs in self.split.items():
            for x in xs:
                r = self.raw[(f[name], x["example_id"])]
                self.assertEqual(x["label_names"][x["gold_label"]], r["label"])
                if name != "test_all":
                    self.assertEqual(x["prover_label"], r["label"])
        self.assertEqual(len(self.split["test_all"]), 203)

    def test_splits_disjoint_by_story(self):
        tr = {x["story_id"] for x in self.split["train"]}
        self.assertFalse(tr & {x["story_id"] for x in self.split["val"]})
        self.assertFalse(tr & {x["story_id"] for x in self.split["test_all"]})

    def test_nodes_reverified(self):
        """Every node of val, test and the first 150 train items: the premises entail its answer, it is not
        equivalent to the conclusion or its negation, its wording carries the truth, yes / no is balanced."""
        yes = n = 0
        for xs in (self.split["val"], self.split["test"], self.split["train"][:150]):
            for x in xs:
                src = "folio_v2_validation.jsonl" if x["item_id"].startswith("foliotf-test") else "folio_v2_train.jsonl"
                r = self.raw[(src, x["example_id"])]
                prem = [B.parse(p) for p in r["premises-FOL"].split("\n") if p.strip()]
                W = B.World()
                s = B.solver(W.ctx)
                s.add(*[W.to_z3(p) for p in prem])
                zc = W.to_z3(B.parse(r["conclusion-FOL"]))
                for q in x["subqs"]:
                    name, args = q["atom"]
                    atom = W.pred(name, len(args))(*[W.const(a) for a in args])
                    holds = q["truth"] != q["flipped"]
                    self.assertEqual(B.entails(s, atom if holds else z3.Not(atom)), "yes", (x["item_id"], q))
                    self.assertFalse(B.valid(atom == zc) or B.valid(atom == z3.Not(zc)), (x["item_id"], q))
                    self.assertTrue(q["question"].startswith("Is it false that" if q["flipped"] else "Is it true that"))
                    self.assertEqual(q["p_cot"], float(q["truth"]))
                    self.assertEqual((q["depends_on"], q["node_depth"]), ([], 0))
                    yes += q["truth"]
                    n += 1
        self.assertGreater(n, 500)
        self.assertTrue(0.4 < yes / n < 0.6, yes / n)

    def test_wording_does_not_predict_the_answer(self):
        """Per split and per frame (property / relation): as many Yes as No under each wording; every asked atom's
        answer follows its wording; a not-holding atom may be asked in both wordings, a holding atom in one."""
        summ = json.loads((DATA / "summary.json").read_text())
        for name in ("train", "val", "test"):
            qs = [q for x in self.split[name] for q in x["subqs"]]
            for frame in ("has the property", "stands in the relation"):
                for flip in (False, True):
                    g = [q["truth"] for q in qs if q["flipped"] == flip and frame in q["question"]]
                    self.assertLessEqual(abs(2 * sum(g) - len(g)), 1, (name, frame, flip))
            for q in qs:
                self.assertEqual(q["truth"], q["atom_holds"] != q["flipped"])
            for w in ("true", "false"):
                self.assertAlmostEqual(summ["splits"][name][f"yes_given_{w}_wording"][2], 0.5, delta=0.01)
            for x in self.split[name]:
                seen = {}
                for q in x["subqs"]:
                    seen.setdefault((q["atom"][0], tuple(q["atom"][1])), []).append(q)
                for qq in seen.values():
                    self.assertLessEqual(len(qq), 2 if not qq[0]["atom_holds"] else 1)
                    self.assertEqual(len({q["flipped"] for q in qq}), len(qq))

    def test_rebuild_identical(self):
        summ = json.loads((DATA / "summary.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            old = B.OUT, B.TEST, B.TEST_ALL, B.ROOT
            B.OUT, B.TEST, B.TEST_ALL, B.ROOT = d / "student_folio_tf", d / "t.jsonl", d / "ta.jsonl", d
            try:
                B.build(0)
                got = {"train": B.OUT / "train.jsonl", "val": B.OUT / "val.jsonl", "test": B.TEST, "test_all": B.TEST_ALL}
                for k, p in got.items():
                    self.assertEqual(hashlib.md5(p.read_bytes()).hexdigest(), summ["splits"][k]["md5"], k)
            finally:
                B.OUT, B.TEST, B.TEST_ALL, B.ROOT = old


if __name__ == "__main__":
    unittest.main()
