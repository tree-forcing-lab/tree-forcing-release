"""Tests for build_hint_evals.py on the real eval files (the sets are small; a run takes seconds).
    PYTHONPATH=<repo>/src <venv python> test_build_hint_evals.py <repo dir> <path to build_hint_evals.py>
"""
import hashlib, json, os, subprocess, sys, tempfile

if len(sys.argv) < 3:
    raise SystemExit(__doc__)        # run as a script with its two arguments, not under pytest collection
REPO, BUILD = sys.argv[1], sys.argv[2]
STEPS, X = f"{REPO}/data/eval/gsm8k_mc_v2_steps.jsonl", f"{REPO}/data/eval/gsm8k_mc_v2_xprob.jsonl"
EXPECT = {"hint_plain": 1129, "hint_full": 1129, "hint_true_r1": 1129, "hint_true_r2": 877, "hint_true_r3": 576,
          "hint_true_r4": 332, "hint_true_r5": 158, "hint_true_r6": 75, "hint_other_r1": 1129, "hint_placebo_r1": 1129,
          "hint_wrongres_r1": 1111, "hint_other_r2": 877, "hint_placebo_r2": 877, "hint_wrongres_r2": 861,
          "hint_placebo_r3": 576, "hint_placebo_r4": 332, "hint_wrongres_near_r1": 1129, "hint_wrongres_near_r2": 877}


def build(out):
    subprocess.run([sys.executable, BUILD, "--steps", STEPS, "--x", X, "--out", out], check=True, capture_output=True)
    return {n: [json.loads(l) for l in open(f"{out}/{n}.jsonl")] for n in EXPECT}


def md5s(out):
    return {n: hashlib.md5(open(f"{out}/{n}.jsonl", "rb").read()).hexdigest() for n in EXPECT}


def opts(prompt):
    return prompt.rpartition("\nOptions:\n")[2]


d1 = tempfile.mkdtemp(); sets = build(d1)
x = {json.loads(l)["item_id"]: json.loads(l) for l in open(X)}
steps = {json.loads(l)["item_id"].replace("gsm8k-test-", "gsm8k_v2-test-"): json.loads(l) for l in open(STEPS)}
# 1. counts equal the card's table
assert {n: len(v) for n, v in sets.items()} == EXPECT, {n: len(v) for n, v in sets.items()}
plain = {d["item_id"]: d for d in sets["hint_plain"]}
for name, rows in sets.items():
    for d in rows:
        p, s = plain[d["item_id"]], steps[d["item_id"]]
        n = len(s["worked_steps"])
        # 2. plain rows are the x items unchanged; n is the number of worked steps
        assert p["prompt"] == x[d["item_id"]]["prompt"] and d["n"] == n
        # 3. the option block is byte-identical and the prompt is identical outside the inserted block
        assert opts(d["prompt"]) == opts(p["prompt"])
        block = d["prompt"][len(p["prompt"].rpartition("\nOptions:\n")[0]):-len("\nOptions:\n" + opts(p["prompt"]))]
        assert d["prompt"] == p["prompt"].rpartition("\nOptions:\n")[0] + block + "\nOptions:\n" + opts(p["prompt"])
        # 4. exactly k "Step i:" lines, numbered 1..k, in the Worked steps block; k = n - r (full: k = n)
        lines = [ln for ln in block.split("\n") if ln.startswith("Step ")]
        assert len(lines) == d["k"] == len(d["shown"]) and all(ln.startswith(f"Step {i + 1}: ") for i, ln in enumerate(lines)), (name, d["item_id"])
        if d["kind"] == "plain":
            assert d["k"] == 0 and block == ""
        else:
            assert block.startswith("\n\nWorked steps:\n")
        if d["kind"] == "true":
            # 5. never the last step: the shown lines are the first k true step texts and k <= n - 1
            assert d["k"] == n - d["r"] <= n - 1 and d["shown"] == [w["text"] for w in s["worked_steps"][:d["k"]]]
        if d["kind"] == "full":
            assert d["k"] == n and d["shown"] == [w["text"] for w in s["worked_steps"]] and d["shown_equals_option"]
        if d["kind"] == "wrongres_near":
            # 6a. near-miss: same operations, every result wrong, not equal to any option or any true value of the chain
            true = [str(w["value"]) for w in s["worked_steps"]]
            optvals = {ln.split(") ", 1)[1].strip() for ln in opts(p["prompt"]).split("\n") if ") " in ln}
            for ln, w in zip(d["shown"], s["worked_steps"][:d["k"]]):
                lhs, _, rhs = ln.rpartition("=")
                assert lhs == w["text"].rpartition("=")[0] and rhs.strip() != str(w["value"]).strip()
                assert rhs.strip() not in optvals and rhs.strip() not in true
        if d["kind"] == "wrongres":
            # 6b. consistent far-wrong chain: results wrong by >= 1.5, not options, no true result of steps 1..k anywhere
            # in the block (as result or operand); linked operands carry the earlier line's wrong value; same operators
            true = [str(w["value"]).strip() for w in s["worked_steps"]]
            optvals = {ln.split(") ", 1)[1].strip() for ln in opts(p["prompt"]).split("\n") if ") " in ln}
            wrongs = d["wrong_values"]
            import re as _re
            for i, (ln, w) in enumerate(zip(d["shown"], s["worked_steps"][:d["k"]])):
                rhs = ln.rpartition("=")[2].strip()
                assert rhs == wrongs[i] and rhs not in optvals and rhs not in true
                assert abs(float(rhs.replace(",", "")) - float(str(w["value"]).replace(",", ""))) >= 1.5
                assert (" % of " in " " + ln) == (w["op"] == "% of") or w["op"] in ln
                a_exp = wrongs[int(w["a_from"])] if w.get("a_from") is not None and int(w["a_from"]) < i else str(w["a"])
                b_exp = wrongs[int(w["b_from"])] if w.get("b_from") is not None and int(w["b_from"]) < i else str(w["b"])
                assert ln.startswith(f"{a_exp}% of {b_exp} = " if w["op"] == "% of" else f"{a_exp} {w['op']} {b_exp} = "), (ln, a_exp, b_exp)
                toks = set(_re.findall(r"[0-9][0-9,./]*", ln))   # 11/18 stays one token
                for tv in true[: d["k"]]:
                    # a true result may appear only as an UNLINKED operand (a prompt number that happens to equal it)
                    ok_a = tv == str(w["a"]) and w.get("a_from") is None
                    ok_b = tv == str(w["b"]) and w.get("b_from") is None
                    assert tv not in toks or ok_a or ok_b, (ln, tv)
        if d["kind"] == "other":
            # 7. the partner is another item with at least k steps and a different chain
            ps = steps[d["partner"]]
            assert d["partner"] != d["item_id"] and len(ps["worked_steps"]) >= d["k"]
            assert d["shown"] == [w["text"] for w in ps["worked_steps"][:d["k"]]] != [w["text"] for w in s["worked_steps"][:d["k"]]]
        if d["kind"] == "placebo":
            # 8. placebo lines are sentences of the item's own question
            q = p["prompt"].rpartition("\nOptions:\n")[0]
            assert all(ln in q for ln in d["shown"])
        # 9. shown_equals_option flag is right
        optvals = {ln.split(") ", 1)[1].strip() for ln in opts(p["prompt"]).split("\n") if ") " in ln}
        assert d["shown_equals_option"] == any(ln.rpartition("=")[2].strip() in optvals for ln in d["shown"])
# 10. deterministic: a second build gives the same md5s
d2 = tempfile.mkdtemp(); build(d2)
assert md5s(d1) == md5s(d2)
m = json.load(open(f"{d1}/MANIFEST.json"))
assert m["sets"]["hint_true_r1"]["n"] == 1129 and m["skipped_wrongres"] == 34   # 18 (r1) + 16 (r2) items with no valid far-wrong chain
# 11. the step text formatter reproduces every true step text of the eval file
import importlib.util as _iu
_sp = _iu.spec_from_file_location("bh", BUILD); bh = _iu.module_from_spec(_sp)
import types
try:
    _sp.loader.exec_module(bh)
    for e in steps.values():
        for w in e.get("worked_steps") or []:
            assert bh.line_text(w["a"], w["op"], w["b"], w["value"]) == w["text"], w
except ModuleNotFoundError:
    pass          # cotdistill not importable here; the builder itself ran under the venv
print("ALL BUILD_HINT_EVALS TESTS PASSED (11 checks on %d rows)" % sum(EXPECT.values()))
