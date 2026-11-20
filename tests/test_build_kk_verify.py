"""scripts/build_kk_verify.py: the verification control (statement truth under non-candidate assignments)."""
import collections
import itertools
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import build_kk_verify as V  # noqa: E402


def items(n=60):
    out = []
    with open(ROOT / "data/student_kk_tf/train.jsonl") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) == n:
                break
    return out


def brute_truth(text, speakers, roles):
    """Independent evaluation of the generator's statement forms (no _kk_formula): the literal connectives."""
    def atom(s):
        w = s.split()
        assert w[1] == "is" and w[2] == "a", s
        return roles[w[0]] == (w[3] == "knight")
    s = text.rstrip(".")
    if s.startswith("If ") and ", then " in s:
        a, b = s[3:].split(", then ")
        return (not atom(a)) or atom(b)
    if " if and only if " in s:
        a, b = s.split(" if and only if ")
        return atom(a) == atom(b)
    if " and " in s:
        a, b = s.split(" and ")
        return atom(a) and atom(b)
    if " or " in s:
        a, b = s.split(" or ")
        return atom(a) or atom(b)
    return atom(s)


class TestVerify(unittest.TestCase):
    def test_nodes(self):
        seen_forms = collections.Counter()
        for it in items():
            pz = V.parse_puzzle(it)
            nodes = V.verify_nodes(it, pz)
            self.assertEqual(len(nodes), len(it["subqs"]))
            V.check_nodes(it, pz, nodes)
            yes = sum(q["truth"] for q in nodes)
            self.assertLessEqual(abs(2 * yes - len(nodes)), 2)
            pairs = set()
            for q in nodes:
                roles, s, text, w = V.parse_question(q["question"], pz["speakers"])
                self.assertNotIn(roles, pz["forbidden"])                        # not the solution, not an option
                self.assertEqual(set(roles), set(pz["speakers"]))
                try:
                    t = brute_truth(text, pz["speakers"], roles)
                except (AssertionError, ValueError, KeyError):
                    continue                                                    # a form the literal reader skips
                seen_forms[text.split()[0] == "If"] += 1
                self.assertEqual((w == "true") == t, q["truth"], q["question"])
                pairs.add((tuple(sorted(roles.items())), s))
            allpairs = [(tuple(sorted(V.parse_question(q["question"], pz["speakers"])[0].items())),
                         V.parse_question(q["question"], pz["speakers"])[1]) for q in nodes]
            self.assertEqual(len(allpairs), len(set(allpairs)))                 # each (assignment, speaker) once
        self.assertGreater(sum(seen_forms.values()), 500)

    def test_solution_is_gold(self):
        for it in items(30):
            pz = V.parse_puzzle(it)
            sol = pz["solution"]
            sp = pz["speakers"]
            n = sum(all(r[p] == V._kk_val(pz["forms"][p], r) for p in sp)
                    for r in (dict(zip(sp, b)) for b in itertools.product([True, False], repeat=len(sp))))
            self.assertEqual(n, 1)
            self.assertIn(sol, pz["forbidden"])

    def test_deterministic(self):
        it = items(1)[0]
        pz = V.parse_puzzle(it)
        self.assertEqual(V.verify_nodes(it, pz), V.verify_nodes(dict(it), pz))

    def test_built_file(self):
        d = ROOT / "data/student_kk_tf_vrf"
        if not (d / "summary.json").exists():
            self.skipTest("not built")
        s = json.loads((d / "summary.json").read_text())
        self.assertEqual(s["md5"]["data/student_kk_tf_vrf/train.jsonl"], V.md5(d / "train.jsonl"))
        self.assertEqual(s["parse_failures"], 0)
        self.assertLess(abs(s["p_yes_given_true_wording"] - 0.5), 0.01)
        self.assertLess(abs(s["p_yes_given_false_wording"] - 0.5), 0.01)


if __name__ == "__main__":
    unittest.main()
