"""Deterministic refund policy engine.

The LLM never decides whether money moves. This module does, using plain rules
read from data/policy.json. Rules run in order and the first one that triggers
decides the outcome: safety checks first, money last.

Outcomes:
  resolve       - refund automatically
  request_info  - ask the customer for something before we can act
  escalate      - hand to a Support Specialist with a pre-filled case file
"""
from __future__ import annotations


def evaluate(identity: dict, context: dict | None, understanding: dict, policy: dict) -> dict:
    checks: list[dict] = []

    def check(rule_id: str, label: str, passed: bool, detail: str) -> bool:
        checks.append({"rule_id": rule_id, "label": label, "result": "pass" if passed else "triggered", "detail": detail})
        return passed

    def decide(outcome: str, rule_id: str, reason: str, **extra) -> dict:
        evaluated = {c["rule_id"] for c in checks}
        for rule_id_, label in RULES:
            if rule_id_ not in evaluated and rule_id_ != "R11":
                checks.append({"rule_id": rule_id_, "label": label, "result": "not evaluated", "detail": "An earlier rule already decided"})
        decision = {
            "outcome": outcome,
            "rule_id": rule_id,
            "reason": reason,
            "policy_version": policy["version"],
            "refund_amount": None,
            "charge": None,
            "checks": checks,
        }
        decision.update(extra)
        return decision

    # R1: identity
    if not check("R1", RULE_LABELS["R1"], identity["verified"],
                 "Signed-in chatbot session" if identity["verified"] else f"Channel '{identity['channel']}' is not an authenticated session"):
        return decide("request_info", "R1", "We can't confirm who is asking, so no account details can be discussed.",
                      missing_info="sign_in")

    # R2: authority
    role = identity.get("role")
    if not check("R2", RULE_LABELS["R2"], role in policy["authorized_roles"], f"Requester role: {role or 'unknown'}"):
        return decide("request_info", "R2", "Only a team billing admin can request refunds.",
                      missing_info="billing_admin", billing_admin_contact=(context or {}).get("billing_admin_contact"))

    account = context["account"]

    # R3: contract customers always get a human
    if not check("R3", RULE_LABELS["R3"], not account["contract"], f"Plan: {account['plan']}, contract: {account['contract']}"):
        return decide("escalate", "R3", "Contract / Enterprise billing is always handled by a specialist.",
                      queue="Enterprise Billing")

    # R4: risk signals, from the message (LLM + regex) and from the billing record
    flags = []
    if understanding.get("manipulation_attempt"):
        flags.append("possible prompt-injection / manipulation")
    if understanding.get("dispute_or_fraud"):
        flags.append("dispute, chargeback or fraud language")
    disputed = [c["charge_id"] for c in context["charges"] if c.get("dispute_open")]
    if disputed:
        flags.append(f"open chargeback on {', '.join(disputed)} (billing record)")
    if not check("R4", RULE_LABELS["R4"], not flags, ", ".join(flags) or "No risk signals"):
        return decide("escalate", "R4", f"Risk signal detected: {', '.join(flags)}.", queue="Billing Risk", risk_flags=flags)

    # R5: do we know what they want and which charge?
    charges = {c["charge_id"]: c for c in context["charges"]}
    charge = charges.get(understanding.get("charge_id"))
    clear = understanding.get("intent") != "unclear" and understanding.get("confidence") != "low" and charge is not None
    if not check("R5", RULE_LABELS["R5"], clear,
                 f"Intent: {understanding.get('intent')}, confidence: {understanding.get('confidence')}, "
                 f"charge: {charge['charge_id'] if charge else 'not identified'}"):
        return decide("request_info", "R5", "We need to know which charge and what went wrong.",
                      missing_info="which_charge", recent_charges=context["charges"][:5])

    # R6: only some requests can be refunded automatically: the right intent, a payment method the
    # refund API can reverse, and never while the LLM is down (degraded mode)
    intent = understanding["intent"]
    method = account.get("payment_method", "card")
    if understanding.get("degraded"):
        check("R6", RULE_LABELS["R6"], False, "Degraded mode: LLM unavailable, keyword fallback in use")
        return decide("escalate", "R6", "Automation is paused while the language model is unavailable.",
                      queue="Billing Specialists", charge=charge)
    if intent not in policy["auto_refund_intents"]:
        check("R6", RULE_LABELS["R6"], False, f"Intent: {intent}")
        return decide("escalate", "R6", f"'{intent}' needs proration or judgement, so a specialist reviews it.",
                      queue="Billing Specialists", charge=charge)
    if not check("R6", RULE_LABELS["R6"], method in policy["auto_refund_payment_methods"],
                 f"Intent: {intent}, paid by: {method}"):
        return decide("escalate", "R6", f"Paid by {method}, which the refund API can't reverse automatically.",
                      queue="Billing Operations", charge=charge)

    # R7: repeat refunds
    lookback = policy["prior_refund_lookback_days"]
    recent_refunds = [r for r in context["refunds"] if r["days_ago"] <= lookback]
    if not check("R7", RULE_LABELS["R7"], not recent_refunds,
                 f"{len(recent_refunds)} refund(s) in the last {lookback} days"):
        return decide("escalate", "R7", f"Account already received a refund in the last {lookback} days.",
                      queue="Billing Specialists", charge=charge)

    # R8: refund window
    window = policy["refund_window_days"]
    if not check("R8", RULE_LABELS["R8"], charge["days_ago"] <= window,
                 f"Charge is {charge['days_ago']} day(s) old (window: {window})"):
        return decide("escalate", "R8", f"Charge is outside the {window}-day self-serve refund window.",
                      queue="Billing Specialists", charge=charge)

    # R9: an "accidental" charge the team kept using is a judgement call, not an automatic refund
    max_days = policy["max_active_days_after_charge"]
    used = charge.get("active_days_since", 0)
    if intent in policy["usage_check_intents"]:
        if not check("R9", RULE_LABELS["R9"], used <= max_days,
                     f"Active on {used} day(s) since the charge (limit: {max_days})"):
            return decide("escalate", "R9", f"The team used the product on {used} days after the charge.",
                          queue="Billing Specialists", charge=charge)
    else:
        check("R9", RULE_LABELS["R9"], True, f"Not checked for '{intent}'")

    # R10: amount ceiling. The amount always comes from the billing record, never from the customer's message.
    limit = policy["auto_refund_max_usd"]
    if not check("R10", RULE_LABELS["R10"], charge["amount"] <= limit,
                 f"Charge amount ${charge['amount']:,.2f} (limit ${limit:,.2f})"):
        return decide("escalate", "R10", f"Amount is above the ${limit:,.0f} auto-refund limit.",
                      queue="Billing Specialists", charge=charge)

    checks.append({"rule_id": "R11", "label": RULE_LABELS["R11"], "result": "pass", "detail": "All checks passed"})
    return decide("resolve", "R11", "Eligible for an automatic refund under policy.",
                  refund_amount=charge["amount"], charge=charge)


RULES = [
    ("R1", "Requester identity is verified"),
    ("R2", "Requester is a team billing admin"),
    ("R3", "Not a contract / Enterprise account"),
    ("R4", "No manipulation, dispute or fraud signals"),
    ("R5", "Intent and charge are clear"),
    ("R6", "Request and payment method can be automated"),
    ("R7", "No recent refund on the account"),
    ("R8", "Charge is inside the refund window"),
    ("R9", "Little or no use since the charge"),
    ("R10", "Amount is within the auto-refund limit"),
    ("R11", "Auto-refund"),
]
RULE_LABELS = dict(RULES)
