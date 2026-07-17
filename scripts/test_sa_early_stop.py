#!/usr/bin/env python3
"""Unit tests for progressive SA self-consistency early stopping.

Exercises the REAL scoring path (calculate_sa_score / choose_best_sa_answer)
and the progressive loop (generate_sa_self_consistency_result_progressive) with
lightweight fakes so no GPU / BERTScore model is loaded:

  * FakeLLM  — returns a scripted answer per (round, node), so signatures are
               fully deterministic.
  * FakeScorer — stands in for BERTScorer; returns a constant F1 so thought
               consistency is well-defined without deberta-xlarge-mnli.

Run:  PYTHONNOUSERSITE=1 <ava py> scripts/test_sa_early_stop.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from AVA.operate import (  # noqa: E402
    _sa_round_signature,
    calculate_sa_score,
    choose_best_sa_answer,
    generate_sa_self_consistency_result_progressive,
)


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class _Vec:
    def __init__(self, v):
        self.v = v

    def mean(self):
        return _Scalar(self.v)


class FakeScorer:
    """Mimics BERTScorer.score -> (f1, precision, recall) tensors."""

    def __init__(self, f1=0.9):
        self.f1 = f1
        self.calls = 0

    def score(self, text1, text2):
        self.calls += 1
        return _Vec(self.f1), None, None


class FakeLLM:
    """Returns a scripted answer per round/node.

    `schedule[r][i]` is the answer letter node i emits in round r (0-indexed).
    Each batch_generate_response call is one round; node index is parsed from
    the input prompt text `node-<i>`.
    """

    def __init__(self, schedule):
        self.schedule = schedule
        self.round = 0

    def batch_generate_response(self, prompts):
        answers = self.schedule[self.round]
        out = []
        for p in prompts:
            idx = int(p["text"].split("-")[1])
            ans = answers[idx]
            out.append(json.dumps({"Answer": ans, "Analysis": f"reason-{ans}-r{self.round}-n{idx}"}))
        self.round += 1
        return out


def make_tree(num_nodes):
    return [
        {"path": ["Root", "SA", str(i)], "input_prompt": f"node-{i}"}
        for i in range(num_nodes)
    ]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_signature_helper():
    print("test_signature_helper")
    ranked = [
        {"path": ["Root", "SA", "0"], "final_score": {"A": 0.9}},
        {"path": ["Root", "SA", "1"], "final_score": {"B": 0.5}},
        {"path": ["Root", "SA", "2"], "final_score": {"A": 0.4}},
    ]
    sig, diag = _sa_round_signature(ranked)
    check(sig == (("Root", "SA", "0"), "A", ("Root", "SA", "1"), "B"),
          "signature picks top1 and highest different-answer node")
    check(diag["best_different_answer"] == "B", "best-different answer is B")

    ranked_same = [
        {"path": ["Root", "SA", "0"], "final_score": {"A": 0.9}},
        {"path": ["Root", "SA", "1"], "final_score": {"A": 0.5}},
    ]
    sig2, _ = _sa_round_signature(ranked_same)
    check(sig2 == (("Root", "SA", "0"), "A", None, None),
          "best-different is None when every node agrees with top1")


def test_early_stop_triggers():
    print("test_early_stop_triggers")
    # Constant signature every round -> should stop as soon as stable_count hits
    # stability_rounds. node0->A, node1->B, node2->A, forever.
    schedule = [["A", "B", "A"] for _ in range(8)]
    tree = make_tree(3)
    scorer = FakeScorer()
    _, meta = generate_sa_self_consistency_result_progressive(
        tree, FakeLLM(schedule), self_consistency_num=8,
        max_rounds=8, stability_rounds=4, early_stop=True, scorer=scorer,
    )
    check(meta["early_stopped"] is True, "early_stopped flag set")
    check(meta["executed_rounds"] == 4, "stops at round == stability_rounds (4)")
    check(meta["total_responses"] == 4 * 3, "total responses = executed_rounds * nodes = 12")
    check(meta["full_budget_responses"] == 8 * 3, "full budget = 24")
    check(abs(meta["saving_ratio"] - (24 - 12) / 24) < 1e-9, "saving_ratio = 0.5")
    counts = [r["stable_count"] for r in meta["rounds"]]
    check(counts == [1, 2, 3, 4], f"stable_count increments 1..4, got {counts}")
    check(len(tree[0]["responses"]) == 4, "each node accumulated 4 responses")


def test_signature_change_resets():
    print("test_signature_change_resets")
    # round0: [A,B]  -> top1=node0/A, best-diff=node1/B
    # round1: [A,A]  -> node1 now agrees on A, best-diff becomes None => signature changes
    # rounds 2+: [A,A] -> stable again
    schedule = [["A", "B"], ["A", "A"], ["A", "A"], ["A", "A"], ["A", "A"]]
    tree = make_tree(2)
    _, meta = generate_sa_self_consistency_result_progressive(
        tree, FakeLLM(schedule), self_consistency_num=8,
        max_rounds=5, stability_rounds=3, early_stop=True, scorer=FakeScorer(),
    )
    rounds = meta["rounds"]
    sig0 = tuple(rounds[0]["signature"][1:2]), rounds[0]["signature"][2]
    check(rounds[0]["stable_count"] == 1, "round1 stable_count starts at 1")
    check(rounds[0]["signature"] != rounds[1]["signature"], "signature changes at round 2")
    check(rounds[1]["stable_count"] == 1, "stable_count RESETS to 1 on signature change")
    check(rounds[2]["stable_count"] == 2, "stable_count resumes counting after reset")
    # stability_rounds=3 -> needs stable_count 3 at round4 (index 3)
    check(meta["executed_rounds"] == 4 and meta["early_stopped"],
          "stops one round after 3 consecutive stable signatures")


def test_disabled_runs_full_budget():
    print("test_disabled_runs_full_budget")
    schedule = [["A", "B", "A"] for _ in range(8)]
    tree = make_tree(3)
    _, meta = generate_sa_self_consistency_result_progressive(
        tree, FakeLLM(schedule), self_consistency_num=8,
        max_rounds=8, stability_rounds=4, early_stop=False, scorer=FakeScorer(),
    )
    check(meta["early_stopped"] is False, "no early stop when disabled")
    check(meta["executed_rounds"] == 8, "runs all max_rounds")
    check(meta["total_responses"] == 24, "uses full budget (24)")
    check(meta["saving_ratio"] == 0.0, "saving_ratio 0 when full budget consumed")
    check(len(tree[0]["responses"]) == 8, "each node has 8 accumulated responses")


def test_reuses_real_scoring():
    print("test_reuses_real_scoring")
    # Sanity: the accumulated responses re-scored by the REAL functions produce
    # a sensible top1. node0 unanimously A should win.
    schedule = [["A", "B"], ["A", "C"], ["A", "D"], ["A", "B"]]
    tree = make_tree(2)
    tree, _ = generate_sa_self_consistency_result_progressive(
        tree, FakeLLM(schedule), self_consistency_num=8,
        max_rounds=4, stability_rounds=99, early_stop=True, scorer=FakeScorer(),
    )
    scored = calculate_sa_score(
        [{"path": n["path"], "responses": n["responses"]} for n in tree],
        scorer=FakeScorer(),
    )
    ranked = choose_best_sa_answer(scored)
    top_answer = list(ranked[0]["final_score"].keys())[0]
    check(top_answer == "A", "unanimous node0 answer A ranks first via real scoring")


def main():
    tests = [
        test_signature_helper,
        test_early_stop_triggers,
        test_signature_change_resets,
        test_disabled_runs_full_budget,
        test_reuses_real_scoring,
    ]
    for t in tests:
        t()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
