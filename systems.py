"""Mock versions of the systems a Support Specialist touches today.

In production each class would wrap a real API (identity provider, billing
provider, Zendesk). Here they read JSON fixtures so the demo runs offline and
contains no real customer data.
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[0]}•••@{domain}"


class AccountService:
    """Who is this person, which team are they on, and what can they do?"""

    def __init__(self, accounts: list[dict]):
        self._accounts = accounts

    def find_user(self, email: str) -> tuple[dict | None, dict | None]:
        for account in self._accounts:
            for user in account["users"]:
                if user["email"].lower() == email.strip().lower():
                    return user, account
        return None, None

    def account_summary(self, account: dict) -> dict:
        """Account metadata only. Billing data comes from BillingSystem."""
        admins = [u for u in account["users"] if u["role"] == "billing_admin"]
        return {
            "account_id": account["account_id"],
            "team_name": account["team_name"],
            "plan": account["plan"],
            "billing_interval": account["billing_interval"],
            "contract": account["contract"],
            "seats": account["seats"],
            "payment_method": account.get("payment_method", "card"),
            "billing_admin_contact": _mask_email(admins[0]["email"]) if admins else None,
        }


class BillingSystem:
    """Charges, refunds, and the one write action in the whole workflow."""

    def __init__(self, accounts: list[dict], today: date):
        self._accounts = {a["account_id"]: a for a in accounts}
        self._today = today
        self._refunds_by_key: dict[str, dict] = {}

    def _with_dates(self, rows: list[dict]) -> list[dict]:
        out = []
        for row in rows:
            row = dict(row)
            row["date"] = (self._today - timedelta(days=row["days_ago"])).isoformat()
            out.append(row)
        return out

    def get_charges(self, account_id: str) -> list[dict]:
        return self._with_dates(self._accounts[account_id]["charges"])

    def get_refunds(self, account_id: str) -> list[dict]:
        return self._with_dates(self._accounts[account_id]["refunds"])

    def create_refund(self, account_id: str, charge_id: str, amount: float, idempotency_key: str) -> dict:
        """Idempotent: the same key always returns the same refund, never a second one."""
        if idempotency_key in self._refunds_by_key:
            existing = dict(self._refunds_by_key[idempotency_key])
            existing["replayed"] = True
            return existing
        charge = next(c for c in self._accounts[account_id]["charges"] if c["charge_id"] == charge_id)
        if amount > charge["amount"]:
            raise ValueError("Refund amount cannot exceed the original charge")
        refund = {
            "refund_id": f"re_{uuid.uuid4().hex[:10]}",
            "charge_id": charge_id,
            "amount": round(amount, 2),
            "status": "succeeded",
            "idempotency_key": idempotency_key,
            "replayed": False,
        }
        self._refunds_by_key[idempotency_key] = refund
        self._accounts[account_id]["refunds"].append(
            {"refund_id": refund["refund_id"], "charge_id": charge_id, "amount": refund["amount"], "days_ago": 0}
        )
        return dict(refund)


class Zendesk:
    """System of record for every conversation, whether or not a human touches it."""

    def __init__(self):
        self.tickets: dict[str, dict] = {}

    def upsert_ticket(self, case_id: str, requester: str, channel: str, subject: str) -> dict:
        ticket = self.tickets.setdefault(case_id, {
            "ticket_id": f"ZD-{10000 + len(self.tickets) + 1}",
            "requester": requester,
            "channel": channel,
            "subject": subject,
            "status": "new",
            "group": None,
            "tags": [],
            "comments": [],
        })
        return ticket

    def add_comment(self, case_id: str, body: str, public: bool, author: str) -> None:
        self.tickets[case_id]["comments"].append({"public": public, "author": author, "body": body})

    def update(self, case_id: str, *, status: str | None = None, group: str | None = None, tags: list[str] | None = None) -> None:
        ticket = self.tickets[case_id]
        if status:
            ticket["status"] = status
        if group:
            ticket["group"] = group
        if tags:
            ticket["tags"] = sorted(set(ticket["tags"]) | set(tags))


class Systems:
    """One bundle of fresh mock systems. Evals build a new one per case."""

    def __init__(self, today: date | None = None):
        accounts = copy.deepcopy(load_json("accounts.json")["accounts"])
        self.data = accounts            # shared with the services below; the Admin tab edits it in place
        self.today = today or date.today()
        self.accounts = AccountService(accounts)
        self.billing = BillingSystem(accounts, self.today)
        self.zendesk = Zendesk()
        self.policy = load_json("policy.json")
