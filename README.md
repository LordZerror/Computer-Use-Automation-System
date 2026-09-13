# Computer-Use Automation System

A backend integration layer that lets an AI agent operate a legacy-style back-office
UI: an LLM drives the app once ("discovery"), the successful run is recorded as a
typed, versioned **capability artifact**, and that artifact then **replays
deterministically** — no LLM in the loop — as the path a production AI agent would
trigger. See [REPORT.md](REPORT.md) for the design write-up.

Target surface: [saucedemo.com](https://www.saucedemo.com/) (a public demo site),
standing in for a legacy bank back-office screen. Demo goal: *"log in, add the
Sauce Labs Backpack to the cart, fill out checkout, and reach the checkout
overview page."*

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

You need a [Groq](https://console.groq.com/) API key for the discovery step only
(replay never calls an LLM):

```bash
export GROQ_API_KEY=sk-...
```

Run the tests (no key, no network, no LLM required):

```bash
pytest tests/ -q
```

## Demo path

**1. Discovery** — an LLM-driven run against the live site, saved as a capability artifact:

```bash
python -m src.cli discover \
  --goal "Log in, add the Sauce Labs Backpack to the cart, fill out checkout with the given name/zip, and reach the checkout overview page." \
  --target-url https://www.saucedemo.com/ \
  --params username=standard_user password=secret_sauce first_name=John last_name=Doe zip_code=94107 \
  --sensitive password \
  --app-id saucedemo \
  --capability-id add_to_cart_checkout \
  --checkpoint-kind text_present --checkpoint-text "Checkout: Overview" \
  --run-id discovery1
```

This opens a real (visible) browser window, runs the agent loop, and on success
writes `artifacts/add_to_cart_checkout.json` plus a full log + screenshots to
`evidence/discovery-discovery1/`.

**2. Replay** — deterministic, no LLM, using the saved artifact:

```bash
python -m src.cli replay \
  --artifact artifacts/add_to_cart_checkout.json \
  --params username=standard_user password=secret_sauce first_name=John last_name=Doe zip_code=94107 \
  --run-id replay1
```

Prints a structured result (`success` + typed outputs) and writes evidence to
`evidence/replay-replay1/`.

**3. Replay against an exceptional state** — same artifact, wrong password, to show
the business-outcome path (not a crash):

```bash
python -m src.cli replay \
  --artifact artifacts/add_to_cart_checkout.json \
  --params username=standard_user password=WRONG first_name=John last_name=Doe zip_code=94107 \
  --run-id replay-badlogin
```

Returns `{"status": "business_outcome", "outcome_code": "invalid_login", ...}`.

**4. Stretch goal — agent-facing capability API:**

```bash
python -m src.cli serve
curl http://localhost:8000/capabilities
curl -X POST http://localhost:8000/capabilities/add_to_cart_checkout/invoke \
  -H "Content-Type: application/json" \
  -d '{"username":"standard_user","password":"secret_sauce","first_name":"John","last_name":"Doe","zip_code":"94107"}'
```

## Running without live services

`pytest tests/` runs fully offline (Chromium + `page.set_content`, no network, no
LLM). `replay` needs network access to the target site but never calls Groq;
`discover` needs both network and `GROQ_API_KEY`.

## Project layout

```
src/agent/        perception (a11y-tree snapshot) + Groq tool-calling loop
src/artifact/      capability schema, recorder (transcript -> artifact), store
src/replay/        deterministic executor, ranked locator resolution, error taxonomy
src/safety/        allowlist, risk classification, redaction
src/escalation/    human-in-the-loop pause / live handoff / resume
src/evidence/      structured JSONL logging + screenshots
src/capabilities/  stretch goal: artifacts as a callable capability API
config/            allowlist.yaml, error_signatures.yaml
artifacts/         saved capability artifacts
evidence/          discovery + replay run evidence (checked in)
```
