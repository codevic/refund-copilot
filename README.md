# Refund Copilot

A working prototype of an AI-assisted workflow for billing refund requests, built for the Support Engineer, AI Infrastructure take-home.

**Core idea:** *the AI reads the request, code decides, people handle the edge cases.*
The language model reads the customer's message and writes the reply. It never decides whether money moves.
A versioned, deterministic policy engine makes that decision. Anything risky, ambiguous or high-value goes to a Support Specialist, who gets a case file that is already filled in.

All data is synthetic. Billing, account and Zendesk systems are mocked with JSON fixtures, so nothing here touches a real system.

---

## Quick start (about 5 minutes)

**You need:** Python 3.10 or newer. An Anthropic API key is optional: without one, the app runs in offline mode (keyword rules and templates), which is enough to explore every screen.

### macOS / Linux

```bash
cd refund-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."      # optional; skip for offline mode
streamlit run app.py
```

### Windows (PowerShell)

```powershell
cd refund-copilot
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."      # optional; skip for offline mode
streamlit run app.py
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that window.

The app opens at **http://localhost:8501**. The sidebar shows **Live (Claude)** when a key is found, otherwise **Offline**. You can switch at any time.

### Model

The default model is `claude-opus-5`. To use another one, set `REFUND_COPILOT_MODEL` before starting the app, for example `export REFUND_COPILOT_MODEL=claude-sonnet-5`. Each request makes two model calls (understand, then draft the reply) and takes about 9 seconds end to end with the default model.

### Hosted demo (Streamlit Community Cloud)

The app can run on Streamlit Community Cloud without code changes. In the app's **Secrets** settings add:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."        # use a dedicated key with a spend limit
DEMO_PASSCODE = "choose-a-passcode"     # live Claude stays off until a viewer enters it
```

With `DEMO_PASSCODE` set, anyone with the link can use the whole demo in offline mode, and live Claude calls only switch on after the passcode is entered in the sidebar. Locally, leave `DEMO_PASSCODE` unset and live mode works as soon as a key is present.

---

## Try the demo (about 10 minutes)

Pick a customer in the sidebar, then click the **Send** suggestion under the chat. The right side fills in the case file: the outcome, the policy checklist, what the AI read, the billing context and a full audit trace (purple = LLM step, green = code, grey = system, orange = human).

| Scenario | What happens | What to look for |
|---|---|---|
| **A · Accidental renewal** | Auto-refund of $432 (R11) | Every check passes; the amount comes from the billing record |
| **B · Editor, not an admin** | Asks for the billing admin (R2) | Billing data is never read for an unauthorised requester |
| **C · Enterprise contract** | Escalated (R3) | Case brief with a suggested next step in the Specialist queue |
| **D · Prompt injection** | Escalated to Billing Risk (R4) | "Customer asked for $5,000.00. Ignored." The reply doesn't reveal the flag |
| **E · Vague request** | Asks which charge (R5), then resolves after the follow-up | Send the suggested follow-up; the same ticket continues |
| **F · Email ticket** | Asks the customer to sign in (R1) | Nothing about the account is discussed over email |
| **G · Chargeback already open** | Escalated to Billing Risk (R4) | The message looks normal; only the billing record shows the dispute |

Then try the other tabs:

- **Specialist queue:** escalated cases with a pre-filled brief. Approve (optionally a partial amount) or decline with a reason; the system executes, replies and records it.
- **Zendesk (mock):** every case has a ticket, a public reply and an internal note.
- **Evals:** run the golden set in the current mode (see below).
- **Admin:** change the policy or billing data live. For example, lower the auto-refund limit from 500 to 300, click **Preview impact on the golden set**, then **Publish**. Mark an account as having an open chargeback and rerun its customer.

Tips: **New conversation (same customer)** starts a fresh case after you change something. **Reset demo data** in the sidebar restores the original policy, billing data and tickets.

---

## Run the evals from the command line

```bash
python evals.py          # offline keyword baseline: free, instant, deterministic
python evals.py --live   # with Claude: 23 cases x 2 model calls, about a minute
```

`data/golden_cases.json` holds 23 labelled cases: 19 core scenarios (including open chargeback, heavy use after renewal and invoice payment) plus 4 paraphrase and trap cases.

| Metric | Why it matters |
|---|---|
| False auto-refunds (target 0) | The costly error: money out and policy broken |
| Outcome accuracy / rule accuracy | The right action, for the right reason |
| Intent accuracy | Quality of the understanding step |
| Reply checker pass rate | How often the model's draft is safe to send as written |
| Idempotency check | A retry never creates a second refund |

Results from our runs:

| | False auto-refunds | Outcome | Rule | Intent |
|---|---|---|---|---|
| Keyword baseline (offline) | 1 | 83% | 83% | 75% |
| Claude (`claude-opus-5`) | 0 | 100% | 96% | 94% |

The keyword baseline fails the four paraphrase and trap cases, including one **false auto-refund** (P04: the customer writes "please don't refund anything"). Claude's one rule miss (G09) still escalates, and is arguably a labelling question. Live results can vary slightly between runs.

---

## How it works

| # | Step | Who does it | Notes |
|---|------|-------------|-------|
| 1 | Intake | System | Chatbot and email requests become one `Case` and one Zendesk ticket |
| 2 | Verify identity and authority | Code | Only a signed-in billing admin is authorised |
| 3 | Gather context | Code | Plan, payment method, charges (with dispute status and usage since the charge) and past refunds. Nothing is read before verification (least privilege) |
| 4 | Understand | LLM | Fills a structured form: intent (fixed list), charge id (from the account's own charges only), risk flags |
| 5 | Decide | Code | Ordered rules R1–R11 from `data/policy.json`. The first rule that triggers decides the outcome |
| 6 | Act | System | Idempotent refund, or route or park the ticket |
| 7 | Reply | LLM + code | The model drafts, a deterministic checker verifies, and a safe template is sent if the check fails |
| 8 | Record | System | Public reply, internal case brief and a full trace on every case |

### Policy rules (illustrative thresholds)

R1 identity verified → R2 billing admin → R3 not a contract account → R4 no manipulation or fraud signal in the message and no open chargeback in billing → R5 intent and charge clear → R6 request type and payment method can be automated (and the model is available) → R7 no refund in the last 12 months → R8 charge within 14 days → R9 at most 3 active days since the charge → R10 amount ≤ $500 → R11 auto-refund.

The billing fields behind these rules are read by code only. The model sees charge id, date, type, description and amount, and nothing about the customer.

### Safety properties

- **The refund amount only ever comes from the billing record**, never from the message. An injected "$5,000" is recorded and ignored.
- **Prompt injection is checked twice.** The model flags it, and a regex backstop flags it independently. Either flag escalates the case.
- **Degraded mode:** if the model is unavailable, the workflow still runs on keyword rules but **stops auto-refunding** and escalates instead.
- **An idempotency key per case and charge,** so retries never refund twice.
- **Reply guardrail:** no amounts outside the decision, no refund promised unless one was issued, and no internal rule ids or risk flags shown to the customer.
- **No auto-deny:** every decline is a specialist's decision.

---

## Project layout

```
app.py        Streamlit demo: chat and case file, specialist queue, mock Zendesk, evals, admin
pipeline.py   Orchestrator (the eight steps) and specialist actions
policy.py     Deterministic rules engine (R1–R11)
llm.py        Claude calls with structured output, offline fallback, reply checker
systems.py    Mock account service, billing system (idempotent refunds) and Zendesk
evals.py      Golden-set runner, scorecard and policy impact preview
data/         accounts.json, policy.json, golden_cases.json (all synthetic)
```

It's plain Python calling the Anthropic SDK directly; no agent framework. The workflow is a fixed sequence with the model in two slots, so every step stays visible and testable.

## Limitations

This is a prototype, not production code: the systems are mocks, state lives in the Streamlit session, the policy thresholds are placeholders, and 23 eval cases is a smoke test rather than a benchmark.

## Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar says "No Anthropic credentials found" | The key isn't set in the terminal that started the app. Set it in that same terminal and restart `streamlit run app.py` |
| Replies say the model is unavailable, or the trace shows "API error 401" | The key is wrong or has no access to the model. Check the key, or set `REFUND_COPILOT_MODEL` to a model your key can use |
| `TypeError` about `st.tabs` or `width` | Streamlit is too old: `pip install -U -r requirements.txt` |
| Port 8501 is busy | `streamlit run app.py --server.port 8502` |
| Windows: `pip install` fails with "No such file or directory" deep inside `site-packages` | The folder path is too long for Windows. Unzip to a short path such as `C:\refund-copilot` and create the venv there |
| Anything looks stuck | Sidebar → **Reset demo data** |
