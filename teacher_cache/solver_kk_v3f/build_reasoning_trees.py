#!/usr/bin/env python3
"""Solve the local Knights and Knaves batches and build checkable proof trees.

Uses only Python's standard library. Proof text is an explicit mathematical
certificate, not a transcript of a language model's private reasoning.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path


ATOM = r"([A-Z][a-z]+) is a (knight|knave)"
NONE_OPTION: dict = {}   # item_id -> letter of a "None of the above" option (v3f augmented items)
NODE_FIELDS = {
    "id", "type", "depends_on", "question", "answer", "opposite",
    "status", "initial_answer", "quote",
}


@dataclass(frozen=True)
class Expression:
    op: str
    atoms: tuple[tuple[str, bool], ...]

    def evaluate(self, roles: dict[str, bool]) -> bool:
        values = [roles[name] == positive for name, positive in self.atoms]
        if self.op == "atom":
            return values[0]
        left, right = values
        if self.op == "and":
            return left and right
        if self.op == "or":
            return left or right
        if self.op == "if":
            return not left or right
        if self.op == "iff":
            return left == right
        raise ValueError(self.op)


def parse_expression(text: str) -> Expression:
    text = text.removesuffix(".")
    match = re.fullmatch(ATOM, text)
    if match:
        return Expression("atom", ((match[1], match[2] == "knight"),))
    for separator, op in [(" and ", "and"), (" or ", "or"),
                          (" if and only if ", "iff")]:
        match = re.fullmatch(ATOM + re.escape(separator) + ATOM, text)
        if match:
            return Expression(op, ((match[1], match[2] == "knight"),
                                   (match[3], match[4] == "knight")))
    match = re.fullmatch("If " + ATOM + ", then " + ATOM, text)
    if match:
        return Expression("if", ((match[1], match[2] == "knight"),
                                  (match[3], match[4] == "knight")))
    raise ValueError(f"Unrecognized statement: {text!r}")


@dataclass
class Puzzle:
    item_id: str
    prompt: str
    source_batch: str
    names: list[str]
    texts: dict[str, str]
    expressions: dict[str, Expression]
    options: dict[str, dict[str, bool]]

    @classmethod
    def read(cls, record: dict, source_batch: str) -> Puzzle:
        assert set(record) == {"item_id", "prompt"}, record["item_id"]
        prompt = record["prompt"]
        statements = re.findall(r'^- ([A-Z][a-z]+) says: "([^"\n]+)"$',
                                prompt, re.MULTILINE)
        names = [name for name, _ in statements]
        population = int(re.search(r"Each of the (\d+) inhabitants", prompt)[1])
        assert population == len(names) == len(set(names))
        texts = dict(statements)
        expressions = {name: parse_expression(text) for name, text in statements}
        assert all(name in names for expr in expressions.values()
                   for name, _ in expr.atoms)
        options = {}
        option_lines = re.findall(r"^([A-Z])\) (.+)$", prompt, re.MULTILINE)
        none_letter = None
        for letter, line in option_lines:
            if line.strip() == "None of the above":      # augmented items (data/student_v3f): no listed assignment holds
                none_letter = letter
                continue
            entries = re.findall(r"([A-Z][a-z]+): (knight|knave)", line)
            assert line == ", ".join(f"{name}: {role}" for name, role in entries)
            assert [name for name, _ in entries] == names
            options[letter] = {name: role == "knight" for name, role in entries}
        assert len(options) + (none_letter is not None) in (6, 7)
        NONE_OPTION[record["item_id"]] = none_letter
        return cls(record["item_id"], prompt, source_batch, names, texts,
                   expressions, options)


def independent_models(puzzle: Puzzle) -> tuple[list[dict[str, bool]], int]:
    """Independently evaluate the original wording without using proof gates."""
    models = []
    tested = 0
    for bits in itertools.product((False, True), repeat=len(puzzle.names)):
        roles = dict(zip(puzzle.names, bits))
        valid = True
        for speaker, text in puzzle.texts.items():
            terms = re.findall(ATOM, text)
            values = [roles[name] == (role == "knight") for name, role in terms]
            if text.startswith("If "):
                truth = (not values[0]) or values[1]
            elif " if and only if " in text:
                truth = values[0] == values[1]
            elif " and " in text:
                truth = all(values)
            elif " or " in text:
                truth = any(values)
            else:
                assert len(values) == 1
                truth = values[0]
            if truth != roles[speaker]:
                valid = False
        tested += 1
        if valid:
            models.append(roles)
    return models, tested


@dataclass(frozen=True)
class Gate:
    speaker: str
    kind: str
    variables: tuple[tuple[str, str], ...]
    rows: tuple[tuple[bool, ...], ...]


def make_gates(puzzle: Puzzle) -> list[Gate]:
    gates = []
    for speaker in puzzle.names:
        role = ("role", speaker)
        truth = ("truth", speaker)
        gates.append(Gate(speaker, "speaker", (role, truth),
                          ((False, False), (True, True))))
        expr = puzzle.expressions[speaker]
        inputs = tuple(dict.fromkeys(("role", name) for name, _ in expr.atoms))
        variables = (truth,) + inputs
        rows = []
        for bits in itertools.product((False, True), repeat=len(inputs)):
            roles = {var[1]: bit for var, bit in zip(inputs, bits)}
            rows.append((expr.evaluate(roles),) + bits)
        gates.append(Gate(speaker, "expression", variables, tuple(rows)))
    return gates


def role_word(value: bool) -> str:
    return "knight" if value else "knave"


def role_list(scope: tuple[tuple[str, bool], ...], *, hypothetical=False) -> str:
    verb = "were" if hypothetical else "is"
    return " and ".join(f"{name} {verb} a {role_word(value)}"
                        for name, value in scope)


class ProofBuilder:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.nodes: list[dict] = []
        self.lines: list[str] = []
        self.predicates: dict[str, tuple] = {}
        self.gates = make_gates(puzzle)
        self.parse_ids = {}
        self.degree = collections.Counter(name for expr in puzzle.expressions.values()
                                          for name, _ in expr.atoms)
        self.max_case_depth = 0
        self.failed_cases = 0
        self.satisfiable_cases = 0
        self.domain = self.add(
            "parse", [],
            f"Is each of the {len(puzzle.names)} inhabitants exactly one of a knight and a knave?",
            "yes", f"Can any of the {len(puzzle.names)} inhabitants be both a knight and a knave?",
            f"Each of the {len(puzzle.names)} inhabitants is exactly one of a knight and a knave.",
            ("domain",),
        )
        self.knight_rule = self.add(
            "parse", [], "Must a knight's statement be true?", "yes",
            "Can a knight's statement be false?",
            "A knight's statement must be true.", ("domain",),
        )
        self.knave_rule = self.add(
            "parse", [], "Must a knave's statement be false?", "yes",
            "Can a knave's statement be true?",
            "A knave's statement must be false.", ("domain",),
        )
        for speaker in puzzle.names:
            text = puzzle.texts[speaker]
            sentence = f'{speaker} is a knight exactly when "{text}" is true.'
            self.parse_ids[speaker] = self.add(
                "parse", [],
                f'Is {speaker} a knight exactly when "{text}" is true?',
                "yes", f'Is {speaker} a knave exactly when "{text}" is true?',
                sentence, ("parse", speaker),
            )

    def add(self, kind, deps, question, answer, opposite, sentence, predicate,
            *, quote=None):
        deps = list(dict.fromkeys(deps))
        assert kind == "parse" or deps
        ident = f"n{len(self.nodes) + 1}"
        quote = quote or sentence
        assert len(quote.split()) <= 30, quote
        self.nodes.append({
            "id": ident, "type": kind, "depends_on": deps,
            "question": question, "answer": answer, "opposite": opposite,
            "status": "direct", "initial_answer": None, "quote": quote,
        })
        self.lines.append(sentence)
        self.predicates[ident] = predicate
        return ident

    def fact(self, var, value, scope, deps, *, verify=False, required=False):
        channel, name = var
        if channel == "role":
            positive = f"{name} be a knight"
            negative = f"{name} be a knave"
            settled = f"{name} {'would be' if scope else 'is'} a {role_word(value)}."
            question = f"Is {name} a knight?"
            opposite = f"Is {name} a knave?"
        elif required:
            text = self.puzzle.texts[name]
            positive = f'{name}\'s role require the statement "{text}" to be true'
            negative = f'{name}\'s role require the statement "{text}" to be false'
            settled = (f"{name}'s role {'would require' if scope else 'requires'} "
                       f"a {'true' if value else 'false'} statement.")
            question = f'Does {name}\'s role require the statement "{text}" to be true?'
            opposite = f'Does {name}\'s role require the statement "{text}" to be false?'
            # The role's required truth value and the evaluated truth value are
            # different judgements, even when a hypothetical case contradicts them.
            var = ("requirement", name)
        else:
            text = self.puzzle.texts[name]
            positive = f'the statement by {name}, "{text}", be true'
            negative = f'the statement by {name}, "{text}", be false'
            settled = (f"{name}'s statement {'would be' if scope else 'is'} "
                       f"{'true' if value else 'false'}.")
            question = f'Is the statement by {name}, "{text}", true?'
            opposite = f'Is the statement by {name}, "{text}", false?'
        if scope:
            question = f"If {role_list(scope, hypothetical=True)}, would {positive}?"
            opposite = f"If {role_list(scope, hypothetical=True)}, would {negative}?"
            sentence = f"If {role_list(scope, hypothetical=True)}, {settled}"
        else:
            sentence = settled
        kind = "verify" if verify else ("case" if scope else "derive")
        return self.add(kind, deps, question, "yes" if value else "no",
                        opposite, sentence, ("fact", scope, var, value), quote=settled)

    def conflict(self, speaker, scope, deps, *, truth_conflict=False):
        assert scope, "An unconditional contradiction would invalidate the puzzle"
        self.failed_cases += 1
        conditional = f"If {role_list(scope, hypothetical=True)}, would "
        if truth_conflict:
            question = conditional + f"{speaker}'s statement be both true and false?"
            opposite = conditional + f"{speaker}'s statement have exactly one truth value?"
            settled = f"{speaker}'s statement would be both true and false, a contradiction."
        else:
            question = conditional + (
                f"{speaker}'s statement contradict the rule that knights tell the truth and knaves lie?"
            )
            opposite = conditional + (
                f"{speaker}'s statement obey the rule that knights tell the truth and knaves lie?"
            )
            settled = f"{speaker}'s statement would contradict the rule for knights and knaves."
        sentence = f"If {role_list(scope, hypothetical=True)}, {settled}"
        return self.add("case", deps, question, "yes", opposite, sentence,
                        ("conflict", scope), quote=settled)

    def evidence(self, state, variables):
        return [state[var][1] for var in variables
                if var in state and state[var][1] is not None]

    def propagate(self, state, scope):
        """One local Boolean consequence per recorded fact; never assume a model."""
        while True:
            changed = False
            for gate in self.gates:
                known = [(j, state[var][0]) for j, var in enumerate(gate.variables)
                         if var in state]
                rows = [row for row in gate.rows
                        if all(row[j] == bit for j, bit in known)]
                parse_id = self.parse_ids[gate.speaker]
                deps = [parse_id] + self.evidence(state, gate.variables)
                if not rows:
                    if gate.kind == "expression":
                        # Record the evaluated statement separately from the contradiction.
                        inputs = gate.variables[1:]
                        input_rows = [row for row in gate.rows
                                      if all(row[j] == bit for j, bit in known if j != 0)]
                        outcomes = {row[0] for row in input_rows}
                        assert len(outcomes) == 1
                        truth = outcomes.pop()
                        actual = self.fact(("truth", gate.speaker), truth, scope,
                                           [parse_id] + self.evidence(state, inputs))
                        previous = state[("truth", gate.speaker)][1]
                        assert previous is not None
                        actual_conflict = self.predicates[previous][2][0] == "truth"
                        return self.conflict(gate.speaker, scope,
                                             [parse_id, previous, actual],
                                             truth_conflict=actual_conflict)
                    # Both the role and the statement's truth already have nodes.
                    return self.conflict(gate.speaker, scope,
                                         deps + [self.knight_rule, self.knave_rule])
                for j, var in enumerate(gate.variables):
                    if var in state:
                        continue
                    values = {row[j] for row in rows}
                    if len(values) == 1:
                        value = values.pop()
                        ident = self.fact(var, value, scope, deps,
                                          required=gate.kind == "speaker" and var[0] == "truth")
                        state[var] = (value, ident)
                        changed = True
                        # Restart with an up-to-date set of premise nodes.
                        break
                if changed:
                    break
            if not changed:
                return None

    def choose(self, state):
        unknown = [name for name in self.puzzle.names if ("role", name) not in state]
        return max(unknown, key=lambda name: (self.degree[name], -self.puzzle.names.index(name)))

    def refute(self, state, scope):
        """Try a case with local propagation and recursively discharge subcases."""
        self.max_case_depth = max(self.max_case_depth, len(scope))
        while True:
            contradiction = self.propagate(state, scope)
            if contradiction:
                return contradiction
            if all(("role", name) in state for name in self.puzzle.names):
                self.satisfiable_cases += 1
                return None
            name = self.choose(state)
            substate = dict(state)
            # An assumption is in each conditional question, never an ungrounded fact.
            substate[("role", name)] = (True, None)
            rejected = self.refute(substate, scope + ((name, True),))
            if rejected is None:
                return None
            ident = self.fact(("role", name), False, scope, [self.domain, rejected])
            state[("role", name)] = (False, ident)

    def solve(self):
        state = {}
        while True:
            contradiction = self.propagate(state, ())
            assert contradiction is None
            if all(("role", name) in state for name in self.puzzle.names):
                break
            name = self.choose(state)
            case_state = dict(state)
            case_state[("role", name)] = (True, None)
            rejected = self.refute(case_state, ((name, True),))
            if rejected is not None:
                value = False
            else:
                # The first case may work. It is still conditional until the other is refuted.
                case_state = dict(state)
                case_state[("role", name)] = (False, None)
                rejected = self.refute(case_state, ((name, False),))
                assert rejected is not None, f"Nonunique puzzle: {self.puzzle.item_id}"
                value = True
            ident = self.fact(("role", name), value, (), [self.domain, rejected])
            state[("role", name)] = (value, ident)
        solution = {name: state[("role", name)][0] for name in self.puzzle.names}
        # Each final verification checks exactly one statement, never a full assignment.
        for speaker in self.puzzle.names:
            expr = self.puzzle.expressions[speaker]
            inputs = list(dict.fromkeys(("role", name) for name, _ in expr.atoms))
            truth = expr.evaluate(solution)
            evaluated = self.fact(("truth", speaker), truth, (),
                                  [self.parse_ids[speaker]] + self.evidence(state, inputs),
                                  verify=True)
            role = role_word(solution[speaker])
            text = self.puzzle.texts[speaker]
            rule = self.knight_rule if solution[speaker] else self.knave_rule
            self.add(
                "verify", [evaluated, state[("role", speaker)][1], rule],
                f'Does the statement by {speaker}, "{text}", obey the rule for {speaker} as a {role}?',
                "yes",
                f'Does the statement by {speaker}, "{text}", contradict the rule for {speaker} as a {role}?',
                f"{speaker}'s statement obeys the rule for {speaker} as a {role}.",
                ("verify", speaker),
            )
        matching = [letter for letter, assignment in self.puzzle.options.items()
                    if assignment == solution]
        if not matching and NONE_OPTION.get(self.puzzle.item_id):
            matching = [NONE_OPTION[self.puzzle.item_id]]
        assert len(matching) == 1, (self.puzzle.item_id, matching)
        return solution, matching[0]


def validate_proof(builder: ProofBuilder):
    """Check each implication over EVERY role assignment, using parents only.

    Crucially, non-parse checks do not silently import the other puzzle statements.
    Hypotheses are antecedents inside the conditional propositions themselves.
    """
    puzzle = builder.puzzle
    assignments = [dict(zip(puzzle.names, bits))
                   for bits in itertools.product((False, True), repeat=len(puzzle.names))]
    universe = (1 << len(assignments)) - 1
    masks = {}
    opposite_checks = 0
    local_dependency_checks = 0
    nonvacuous_checks = 0
    for position, node in enumerate(builder.nodes, 1):
        assert set(node) == NODE_FIELDS
        assert node["id"] == f"n{position}"
        assert node["type"] in {"parse", "derive", "case", "verify"}
        assert node["answer"] in {"yes", "no"}
        assert node["status"] == "direct" and node["initial_answer"] is None
        assert (node["type"] == "parse") == (not node["depends_on"])
        assert node["question"].endswith("?") and node["opposite"].endswith("?")
        assert node["quote"] in builder.lines[position - 1]
        assert len(node["quote"].split()) <= 30
        for text in (node["question"], node["opposite"]):
            assert not re.search(r"fail(?:s|ed)? to|did the solver|the reasoning|[¬∨∧→↔=]", text, re.I)
            assert not re.search(r"\boption\b|all statements|Who is a knight", text, re.I)
            named = set(re.findall(r"\b[A-Z][a-z]+\b", text))
            assert named - {"Is", "If", "Must", "Can", "Does", "Each"} <= set(puzzle.names), named
        predicate = builder.predicates[node["id"]]
        if predicate[0] == "domain":
            mask = universe
        elif predicate[0] in {"parse", "verify"}:
            speaker = predicate[1]
            mask = sum(1 << i for i, roles in enumerate(assignments)
                       if roles[speaker] == puzzle.expressions[speaker].evaluate(roles))
        else:
            scope = predicate[1]
            def holds(roles):
                if any(roles[name] != value for name, value in scope):
                    return True
                if predicate[0] == "conflict":
                    return False
                _, _, (channel, name), value = predicate
                actual = (roles[name] if channel in {"role", "requirement"}
                          else puzzle.expressions[name].evaluate(roles))
                return actual == value
            mask = sum(1 << i for i, roles in enumerate(assignments) if holds(roles))
        if node["type"] != "parse":
            premises = universe
            for parent in node["depends_on"]:
                assert parent in masks, (node["id"], parent)
                premises &= masks[parent]
            assert not (premises & (universe ^ mask)), (
                puzzle.item_id, node["id"], node["question"], node["depends_on"])
            assert premises, ("Vacuous premises", puzzle.item_id, node["id"])
            local_dependency_checks += 1
            nonvacuous_checks += 1
        masks[node["id"]] = mask
        # Surface opposites change only the role, truth, or contradiction predicate.
        if predicate[0] == "parse":
            assert node["opposite"] == node["question"].replace("a knight exactly", "a knave exactly", 1)
        elif predicate[0] == "fact":
            if predicate[2][0] == "role":
                assert node["opposite"] == node["question"].replace("a knight?", "a knave?")
            else:
                assert node["opposite"] == node["question"].removesuffix("true?") + "false?"
            _, scope, (channel, name), _ = predicate
            for roles in assignments:
                if all(roles[person] == value for person, value in scope):
                    positive = (roles[name] if channel in {"role", "requirement"}
                                else puzzle.expressions[name].evaluate(roles))
                    negative = not positive
                    assert positive != negative
        elif predicate[0] == "verify":
            assert node["opposite"] == node["question"].replace(
                "obey the rule", "contradict the rule", 1)
        elif predicate[0] == "conflict":
            scope = predicate[1]
            active = sum(1 << i for i, roles in enumerate(assignments)
                         if all(roles[name] == value for name, value in scope))
            assert not (premises & active)
            if "both true and false?" in node["question"]:
                assert node["opposite"] == node["question"].replace(
                    "be both true and false?", "have exactly one truth value?", 1)
            else:
                assert node["opposite"] == node["question"].replace(
                    "contradict the rule", "obey the rule", 1)
        else:
            assert predicate[0] == "domain"
            assert node["opposite"] in {
                f"Can any of the {len(puzzle.names)} inhabitants be both a knight and a knave?",
                "Can a knight's statement be false?", "Can a knave's statement be true?",
            }
        opposite_checks += 1
    return {
        "nodes": len(builder.nodes),
        "types": dict(collections.Counter(node["type"] for node in builder.nodes)),
        "dependency_entailment_checks": local_dependency_checks,
        "nonvacuous_dependency_checks": nonvacuous_checks,
        "opposite_checks": opposite_checks,
        "max_case_depth": builder.max_case_depth,
        "contradiction_nodes": builder.failed_cases,
        "satisfiable_cases_examined": builder.satisfiable_cases,
        "role_assignments_per_dependency_check": len(assignments),
    }


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reasoning_trees")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    output = base / args.output
    sources = sorted(base.glob("batch_[0-9][0-9].json"))
    puzzles = []
    source_hashes = {}
    for path in sources:
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(records, list)
        puzzles.extend(Puzzle.read(record, path.name) for record in records)
    assert len({p.item_id for p in puzzles}) == len(puzzles)
    if args.limit:
        puzzles = puzzles[:args.limit]
    (output / "trees").mkdir(parents=True, exist_ok=True)
    (output / "proofs").mkdir(exist_ok=True)
    solutions, manifest, checks = [], [], []
    for index, puzzle in enumerate(puzzles, 1):
        builder = ProofBuilder(puzzle)
        solution, answer = builder.solve()
        models, tested = independent_models(puzzle)
        assert len(models) == 1 and models[0] == solution, puzzle.item_id
        check = validate_proof(builder)
        check.update(item_id=puzzle.item_id, unique_solution=True,
                     correct_option=answer, independently_enumerated_assignments=tested)
        checks.append(check)
        tree_name = f"trees/{puzzle.item_id}.json"
        proof_name = f"proofs/{puzzle.item_id}.txt"
        write_json(output / tree_name, {"nodes": builder.nodes})
        proof = "\n".join(builder.lines)
        (output / proof_name).write_text(proof + "\n", encoding="utf-8")
        solutions.append({
            "item_id": puzzle.item_id, "source_batch": puzzle.source_batch,
            "prompt": puzzle.prompt, "answer": answer,
            "roles": {name: role_word(value) for name, value in solution.items()},
            "reasoning": proof, "reasoning_kind": "formal_proof_trace",
            "tree_file": tree_name, "proof_file": proof_name,
        })
        manifest.append({"item_id": puzzle.item_id, "source_batch": puzzle.source_batch,
                         "answer": answer, "tree_file": tree_name,
                         "proof_file": proof_name, "node_count": len(builder.nodes),
                         "tree_sha256": hashlib.sha256((output / tree_name).read_bytes()).hexdigest(),
                         "proof_sha256": hashlib.sha256((output / proof_name).read_bytes()).hexdigest()})
        print(json.dumps({"processed": index, "total": len(puzzles),
                          "item_id": puzzle.item_id, "answer": answer,
                          "nodes": len(builder.nodes)}), flush=True)
    (output / "solutions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in solutions),
        encoding="utf-8")
    write_json(output / "manifest.json", {
        "input_fields": ["item_id", "prompt"],
        "input_batches": len(sources), "puzzle_count": len(puzzles),
        "language": "English", "tree_schema": {"nodes": "list of judgements"},
        "specification": "User-provided fine-grained specification; no node cap.",
        "reasoning_source": "Generated formal proof traces; input contains no raw reasoning.",
        "solver": "Local Boolean propagation with recursive case refutation.",
        "independent_checker": "Exhaustive evaluation of the original statement wording.",
        "source_sha256": source_hashes, "items": manifest,
    })
    type_totals = collections.Counter()
    for check in checks:
        type_totals.update(check["types"])
    report = {
        "status": "passed", "puzzles": len(puzzles),
        "unique_solutions": len(puzzles), "option_matches": len(puzzles),
        "total_nodes": sum(c["nodes"] for c in checks),
        "node_types": dict(type_totals),
        "minimum_nodes": min(c["nodes"] for c in checks),
        "maximum_nodes": max(c["nodes"] for c in checks),
        "dependency_entailment_checks": sum(c["dependency_entailment_checks"] for c in checks),
        "nonvacuous_dependency_checks": sum(c["nonvacuous_dependency_checks"] for c in checks),
        "opposite_checks": sum(c["opposite_checks"] for c in checks),
        "corrected_nodes": 0,
        "correction_note": "Hypothetical contradictions are case deductions, not changes of belief; no corrections were invented.",
        "dependency_semantics": "Each node follows from its listed parents alone, checked on all Boolean role assignments without importing unlisted statements.",
        "case_semantics": "Hypotheses occur explicitly in each conditional question; opposite predicates share the same hypothesis.",
        "truth_semantics": "The truth required by a person's role is separate from the evaluated truth of that person's statement.",
        "items": checks,
    }
    write_json(output / "validation_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "items"}), flush=True)


if __name__ == "__main__":
    main()
