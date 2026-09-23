"""Orchestrator: runs the fixed refund workflow for one case and records every step.

Order of steps (and why):
  1. Intake      - normalise chatbot / email into one Case, open a Zendesk ticket
  2. Verify      - who is asking and are they allowed to ask? (before any billing data is read)
  3. Gather      - plan, charges, past refunds (only for an authorised requester)
  4. Understand  - LLM fills a structured form from the message
  5. Decide      - deterministic policy engine picks resolve / request_info / escalate
  6. Act         - refund (idempotent) or route the ticket
  7. Reply       - LLM drafts, a deterministic checker verifies, template if it fails
  8. Record      - public reply + internal case brief on the ticket, full trace kept
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import llm
import policy as policy_engine
from systems import Systems

SUGGESTED_NEXT_STEP = {
    "R3": "Route to the account team; contract changes follow the order form, not self-serve policy.",
    "R4": "Review for abuse before acting. If a chargeback is open, let the dispute run: refunding now pays out twice.",
    "R6": "Proration or a non-card payment: calculate the amount and refund through the right channel.",
    "R7": "Check the earlier refund. A second refund in 12 months is usually a goodwill decision.",
    "R8": "Outside the window. Default is decline unless there is a product or billing error on our side.",
    "R9": "The team kept using the product after the charge. Offer a downgrade or credit rather than a full refund.",
    "R10": "Otherwise eligible. Approve if the reason holds up; the amount limit is the only reason it is here.",
}


@dataclass
class Case:
    channel: str                      # "chatbot" (signed-in session) or "email" (Zendesk inbound)
    requester_email: str
    conversation: list[dict] = field(default_factory=list)   # {"role": "customer" | "support", "text": str}
    case_id: str = field(default_factory=lambda: f"case_{uuid.uuid4().hex[:8]}")
    run: int = 0
    status: str = "new"               # resolved | awaiting_customer | escalated | closed
    identity: dict | None = None
    context: dict | None = None
    understanding: dict | None = None
    understanding_source: str = ""
    decision: dict | None = None
    refund: dict | None = None
    reply: str | None = None
    reply_source: str = ""
    reply_problems: list[str] = field(default_factory=list)
    human_review: dict | None = None
    trace: list[dict] = field(default_factory=list)

    @property
    def customer_text(self) -> str:
        return "\n".join(m["text"] for m in self.conversation if m["role"] == "customer")


StepCallback = Callable[[dict], None]


def _record(case: Case, on_step: StepCallback | None, step: str, actor: str, summary: str,
            detail: dict | None = None, started: float | None = None) -> None:
    entry = {
        "run": case.run,
        "step": step,
        "actor": actor,
        "summary": summary,
        "detail": detail,
        "ms": int((time.perf_counter() - started) * 1000) if started else 0,
    }
    case.trace.append(entry)
    if on_step:
        on_step(entry)


def run_case(case: Case, sys: Systems, use_llm: bool = True, on_step: StepCallback | None = None) -> Case:
    """(Re)process a case from the full customer conversation so far."""
    case.run += 1
    message = case.customer_text

    # 1. Intake -------------------------------------------------------------
    t = time.perf_counter()
    ticket = sys.zendesk.upsert_ticket(case.case_id, case.requester_email, case.channel,
                                       subject=case.conversation[0]["text"][:60])
    _record(case, on_step, "Intake", "System",
            f"{case.channel} message normalised into {case.case_id}, Zendesk ticket {ticket['ticket_id']}",
            {"channel": case.channel, "ticket": ticket["ticket_id"], "turns": len(case.conversation)}, t)

    # 2. Verify identity and authority -------------------------------------
    t = time.perf_counter()
    user, account = sys.accounts.find_user(case.requester_email)
    verified = case.channel == "chatbot" and user is not None
    case.identity = {
        "channel": case.channel,
        "verified": verified,
        "name": user["name"] if user else None,
        "role": user["role"] if user else None,
        "team": account["team_name"] if account else None,
    }
    authorised = verified and user["role"] in sys.policy["authorized_roles"]
    _record(case, on_step, "Verify identity & authority", "Code",
            f"{'Verified' if verified else 'Not verified'}; role: {case.identity['role'] or 'unknown'}"
            f"{' (authorised)' if authorised else ''}", case.identity, t)

    # 3. Gather context (least privilege) -----------------------------------
    t = time.perf_counter()
    if verified:
        summary = sys.accounts.account_summary(account)
        case.context = {
            "account": summary,
            "billing_admin_contact": summary["billing_admin_contact"],
            "charges": sys.billing.get_charges(account["account_id"]) if authorised else [],
            "refunds": sys.billing.get_refunds(account["account_id"]) if authorised else [],
        }
        note = (f"{summary['plan']} {summary['billing_interval']}, {len(case.context['charges'])} charge(s), "
                f"{len(case.context['refunds'])} past refund(s)") if authorised else \
            "Account metadata only; billing data withheld because requester is not authorised"
    else:
        case.context = None
        note = "Skipped: nothing is read until the requester is verified"
    _record(case, on_step, "Gather context", "Code", note,
            {"account": case.context["account"]} if case.context else None, t)

    # 4. Understand the request ---------------------------------------------
    t = time.perf_counter()
    charges = case.context["charges"] if case.context else []
    if use_llm:
        try:
            u = llm.understand_llm(message, charges)
            case.understanding_source = f"LLM ({llm.MODEL})"
        except llm.LLMUnavailable as e:
            u = llm.understand_offline(message, charges)
            u["degraded"] = True
            case.understanding_source = f"Offline rules (LLM unavailable: {e})"
    else:
        u = llm.understand_offline(message, charges)
        case.understanding_source = "Offline rules"
    backstop = llm.detect_manipulation(message) and not u["manipulation_attempt"]
    if backstop:
        u["manipulation_attempt"] = True
    if u["charge_id"] not in {c["charge_id"] for c in charges}:
        u["charge_id"] = "none"
    case.understanding = u
    _record(case, on_step, "Understand request", "LLM" if case.understanding_source.startswith("LLM") else "Code",
            f"intent={u['intent']}, charge={u['charge_id']}, confidence={u['confidence']}"
            + (", manipulation flagged" if u["manipulation_attempt"] else "")
            + (" (by deterministic backstop)" if backstop else ""),
            {"source": case.understanding_source, **u}, t)

    # 5. Decide ---------------------------------------------------------------
    t = time.perf_counter()
    d = policy_engine.evaluate(case.identity, case.context, u, sys.policy)
    case.decision = d
    _record(case, on_step, "Policy decision", "Code",
            f"{d['rule_id']} -> {d['outcome']}: {d['reason']}",
            {"policy_version": d["policy_version"], "rule_id": d["rule_id"], "outcome": d["outcome"]}, t)

    # 6. Act ------------------------------------------------------------------
    t = time.perf_counter()
    account_id = case.context["account"]["account_id"] if case.context else None
    if d["outcome"] == "resolve":
        key = f"refund:{case.case_id}:{d['charge']['charge_id']}"
        case.refund = sys.billing.create_refund(account_id, d["charge"]["charge_id"], d["refund_amount"], key)
        sys.zendesk.update(case.case_id, status="solved", tags=["refund", "auto_resolved", u["intent"]])
        case.status = "resolved"
        act = f"Refund {case.refund['refund_id']} for ${case.refund['amount']:,.2f} issued (idempotency key {key})"
    elif d["outcome"] == "escalate":
        sys.zendesk.update(case.case_id, status="open", group=d["queue"], tags=["refund", "escalated", d["rule_id"]])
        case.status = "escalated"
        act = f"Ticket routed to '{d['queue']}' with a pre-filled case brief"
    else:
        sys.zendesk.update(case.case_id, status="pending", tags=["refund", "needs_info", d["missing_info"]])
        case.status = "awaiting_customer"
        act = f"Ticket set to pending: waiting for customer ({d['missing_info']})"
    _record(case, on_step, "Act", "System", act, {"refund": case.refund} if case.refund else None, t)

    # 7. Draft and check the reply -------------------------------------------
    _compose_and_send_reply(case, sys, use_llm, on_step)

    # 8. Record the internal brief --------------------------------------------
    sys.zendesk.add_comment(case.case_id, case_brief(case), public=False, author="Refund workflow")
    _record(case, on_step, "Audit", "System",
            f"Internal case brief added to ticket; {len([s for s in case.trace if s['run'] == case.run])} steps recorded")
    return case


def _reply_facts(case: Case) -> dict:
    d = case.decision
    charge = d.get("charge") or {}
    return {
        "outcome": d["outcome"],
        "customer_first_name": (case.identity.get("name") or "there").split()[0],
        "refund_amount": case.refund["amount"] if case.refund else None,
        "refund_reference": case.refund["refund_id"] if case.refund else None,
        "charge_description": charge.get("description"),
        "charge_date": charge.get("date"),
        "charge_amount": charge.get("amount"),
        "missing_info": d.get("missing_info"),
        "billing_admin_contact": d.get("billing_admin_contact"),
        "recent_charges": [{"date": c["date"], "description": c["description"], "amount": c["amount"]}
                           for c in d.get("recent_charges", [])],
    }


def _compose_and_send_reply(case: Case, sys: Systems, use_llm: bool, on_step: StepCallback | None,
                            facts: dict | None = None, override_text: str | None = None) -> None:
    t = time.perf_counter()
    facts = facts or _reply_facts(case)
    facts["sla"] = sys.policy["escalation_sla"]
    if override_text:
        reply, source, problems = override_text, "Specialist template", []
    elif use_llm:
        try:
            reply, source = llm.draft_llm(facts), f"LLM ({llm.MODEL})"
        except llm.LLMUnavailable as e:
            reply, source = llm.draft_template(facts), f"Template (LLM unavailable: {e})"
        problems = llm.check_reply(reply, facts)
        if problems:
            reply, source = llm.draft_template(facts), "Template (LLM draft failed checks)"
    else:
        reply, source, problems = llm.draft_template(facts), "Template", []
    case.reply, case.reply_source, case.reply_problems = reply, source, problems
    case.conversation.append({"role": "support", "text": reply})
    sys.zendesk.add_comment(case.case_id, reply, public=True, author="Figma Support")
    summary = f"Reply sent ({source})"
    if problems:
        summary += f"; draft rejected by checker: {'; '.join(problems)}"
    _record(case, on_step, "Draft & check reply", "LLM" if source.startswith("LLM") else "Code", summary,
            {"source": source, "checker_problems": problems}, t)


def case_brief(case: Case) -> str:
    """What the Support Specialist sees instead of opening four tools."""
    d, u, i = case.decision, case.understanding, case.identity
    lines = [
        f"Outcome: {d['outcome'].upper()} ({d['rule_id']}: {d['reason']})",
        f"Requester: {i['name'] or case.requester_email} | role: {i['role']} | verified: {i['verified']} | via {i['channel']}",
    ]
    if case.context:
        a = case.context["account"]
        lines.append(f"Account: {a['team_name']} | {a['plan']} {a['billing_interval']} | {a['seats']} seats | "
                     f"contract: {a['contract']} | paid by: {a['payment_method']}")
    lines.append(f"Customer says: {u['customer_reason']}")
    lines.append(f"AI read: intent {u['intent']} ({u['confidence']} confidence), charge {u['charge_id']}")
    if u.get("requested_amount"):
        lines.append(f"Customer asked for ${u['requested_amount']:,.2f} (ignored: amounts come from billing records)")
    if d.get("charge"):
        c = d["charge"]
        lines.append(f"Charge in question: {c['charge_id']} | {c['date']} | {c['description']} | ${c['amount']:,.2f} | "
                     f"active {c.get('active_days_since', 0)} day(s) since | dispute open: {c.get('dispute_open', False)}")
    if d.get("risk_flags"):
        lines.append(f"Risk flags: {', '.join(d['risk_flags'])}")
    if u.get("degraded"):
        lines.append("Suggested next step: The language model was unavailable, so handle this as a normal manual review.")
    elif d["rule_id"] in SUGGESTED_NEXT_STEP:
        lines.append(f"Suggested next step: {SUGGESTED_NEXT_STEP[d['rule_id']]}")
    lines.append(f"Policy version: {d['policy_version']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human-in-the-loop actions from the specialist console
# ---------------------------------------------------------------------------

def specialist_approve(case: Case, sys: Systems, charge_id: str, amount: float, note: str,
                       use_llm: bool = True, on_step: StepCallback | None = None) -> Case:
    t = time.perf_counter()
    account_id = case.context["account"]["account_id"]
    charge = next(c for c in case.context["charges"] if c["charge_id"] == charge_id)
    key = f"refund:{case.case_id}:{charge_id}"
    case.refund = sys.billing.create_refund(account_id, charge_id, amount, key)
    case.human_review = {"decision": "approved", "charge_id": charge_id, "amount": amount, "note": note}
    case.status = "resolved"
    sys.zendesk.update(case.case_id, status="solved", tags=["specialist_approved"])
    sys.zendesk.add_comment(case.case_id, f"Specialist approved ${amount:,.2f} on {charge_id}. Note: {note or '-'}",
                            public=False, author="Support Specialist")
    _record(case, on_step, "Specialist review", "Human",
            f"Approved ${amount:,.2f} refund on {charge_id}; refund {case.refund['refund_id']} (key {key})",
            case.human_review, t)
    facts = _reply_facts(case)
    facts.update({"outcome": "resolve", "charge_description": charge["description"], "charge_date": charge["date"],
                  "charge_amount": charge["amount"]})
    _compose_and_send_reply(case, sys, use_llm, on_step, facts=facts)
    return case


def specialist_reject(case: Case, sys: Systems, reason: str, on_step: StepCallback | None = None) -> Case:
    t = time.perf_counter()
    case.human_review = {"decision": "declined", "note": reason}
    case.status = "closed"
    sys.zendesk.update(case.case_id, status="solved", tags=["specialist_declined"])
    _record(case, on_step, "Specialist review", "Human", f"Declined: {reason}", case.human_review, t)
    name = (case.identity.get("name") or "there").split()[0]
    text = (f"Hi {name}, thanks for your patience. A billing specialist reviewed your request and we're not able to "
            f"refund this charge: {reason} If anything here looks wrong, just reply and we'll take another look.\n\n"
            f"Figma Support")
    _compose_and_send_reply(case, sys, False, on_step, override_text=text)
    return case
