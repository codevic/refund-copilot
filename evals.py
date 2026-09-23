"""Golden-set evals for the refund workflow.

Run:  python evals.py            (offline rules, deterministic, free)
      python evals.py --live     (real Claude calls for understand + reply)

Metrics, in order of importance:
  1. False auto-refunds  - cases we refunded that a specialist would not have. Must be 0.
  2. Outcome accuracy    - resolve / request_info / escalate matches the golden label.
  3. Rule accuracy       - the right reason, not just the right outcome.
  4. Intent accuracy     - how well the understanding step reads the message.
  5. Reply checker pass  - drafts that passed the guardrail without falling back to a template.
Plus a system check: replaying a refund with the same idempotency key must not refund twice.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline import Case, run_case
from systems import Systems, load_json


def _run_one(g: dict, use_llm: bool, policy: dict | None = None) -> dict:
    sys = Systems()                                       # fresh mock systems per case, no leakage
    if policy:
        sys.policy = policy
    case = Case(channel=g["channel"], requester_email=g["email"],
                conversation=[{"role": "customer", "text": g["message"]}])
    started = time.perf_counter()
    run_case(case, sys, use_llm=use_llm)
    d, u = case.decision, case.understanding
    return {
        "id": g["id"],
        "title": g["title"],
        "expected": f"{g['expected_outcome']} ({g['expected_rule']})",
        "actual": f"{d['outcome']} ({d['rule_id']})",
        "outcome_ok": d["outcome"] == g["expected_outcome"],
        "rule_ok": d["rule_id"] == g["expected_rule"],
        "intent_expected": g["expected_intent"],
        "intent_actual": u["intent"],
        "intent_ok": None if g["expected_intent"] is None else u["intent"] == g["expected_intent"],
        "false_auto_refund": d["outcome"] == "resolve" and g["expected_outcome"] != "resolve",
        "reply_checker_ok": not case.reply_problems,
        "understood_by": case.understanding_source,
        "seconds": round(time.perf_counter() - started, 1),
    }


def run_golden(use_llm: bool, on_case=None, workers: int = 6, policy: dict | None = None) -> tuple[list[dict], dict]:
    golden = load_json("golden_cases.json")["cases"]
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers if use_llm else 1) as pool:
        futures = [pool.submit(_run_one, g, use_llm, policy) for g in golden]
        for future in as_completed(futures):
            row = future.result()
            results[row["id"]] = row
            if on_case:
                on_case(row)
    rows = [results[g["id"]] for g in golden]

    intent_rows = [r for r in rows if r["intent_ok"] is not None]
    summary = {
        "cases": len(rows),
        "false_auto_refunds": sum(r["false_auto_refund"] for r in rows),
        "outcome_accuracy": sum(r["outcome_ok"] for r in rows) / len(rows),
        "rule_accuracy": sum(r["rule_ok"] for r in rows) / len(rows),
        "intent_accuracy": sum(r["intent_ok"] for r in intent_rows) / len(intent_rows),
        "reply_checker_pass": sum(r["reply_checker_ok"] for r in rows) / len(rows),
        "idempotency_ok": idempotency_check(),
    }
    return rows, summary


def policy_impact(current: dict, candidate: dict) -> list[dict]:
    """Replay the golden set under both policies (offline, deterministic) and list every case whose decision changes.

    Labels were written for the current policy, so this is an impact report, not a pass/fail:
    a change that turns any case into an auto-refund needs sign-off before publishing.
    """
    before, _ = run_golden(False, policy=current)
    after, _ = run_golden(False, policy=candidate)
    changes = []
    for b, a in zip(before, after):
        if b["actual"] != a["actual"]:
            changes.append({
                "id": b["id"],
                "title": b["title"],
                "before": b["actual"],
                "after": a["actual"],
                "new_auto_refund": a["actual"].startswith("resolve") and not b["actual"].startswith("resolve"),
            })
    return changes


def idempotency_check() -> bool:
    sys = Systems()
    first = sys.billing.create_refund("acc_acme", "ch_a1", 432.00, "refund:test:ch_a1")
    retry = sys.billing.create_refund("acc_acme", "ch_a1", 432.00, "refund:test:ch_a1")
    return first["refund_id"] == retry["refund_id"] and retry["replayed"] and len(sys.billing.get_refunds("acc_acme")) == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="use real Claude calls")
    args = parser.parse_args()

    def show(r):
        mark = "PASS" if r["outcome_ok"] and r["rule_ok"] else "FAIL"
        intent = "" if r["intent_ok"] is None else f" intent {'ok' if r['intent_ok'] else 'MISS'} ({r['intent_actual']})"
        print(f"{mark}  {r['id']}  {r['title']:<46} expected {r['expected']:<22} got {r['actual']:<22}{intent}")

    print(f"Running golden set ({'live Claude' if args.live else 'offline rules'})\n")
    _, s = run_golden(args.live, on_case=show)
    print("\nScorecard")
    print(f"  False auto-refunds   {s['false_auto_refunds']}   (target: 0)")
    print(f"  Outcome accuracy     {s['outcome_accuracy']:.0%}")
    print(f"  Rule accuracy        {s['rule_accuracy']:.0%}")
    print(f"  Intent accuracy      {s['intent_accuracy']:.0%}")
    print(f"  Reply checker pass   {s['reply_checker_pass']:.0%}")
    print(f"  Idempotent refunds   {'ok' if s['idempotency_ok'] else 'FAILED'}")


if __name__ == "__main__":
    main()
