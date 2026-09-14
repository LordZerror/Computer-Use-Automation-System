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
(replay never calls an LLM). Either export it, or copy `.env.example` to `.env`
and fill it in (`python-dotenv` auto-loads it — real environment variables
always take precedence over `.env`):

```bash
cp .env.example .env   # then edit .env
# or:
export GROQ_API_KEY=gsk_...
```

Run the tests (no key, no network, no LLM required):

```bash
pytest tests/ -q
```

## Demo path

**1. Discovery** — an LLM-driven run against the live site, saved as a capability artifact:

```bash
python -m src.cli discover \
  --goal "Log in, add the Sauce Labs Backpack to the cart, fill out checkout with the given name/zip, continue to the checkout overview page, and report the order total shown there." \
  --target-url https://www.saucedemo.com/ \
  --params username=standard_user password=secret_sauce first_name=John last_name=Doe zip_code=94107 \
  --sensitive password \
  --app-id saucedemo \
  --capability-id add_to_cart_checkout \
  --checkpoint-kind text_present --checkpoint-text "Checkout: Overview" \
  --max-steps 18 --headless --run-id discovery1
```

Drop `--headless` if you'd rather watch the browser drive itself. On success this writes
`artifacts/add_to_cart_checkout.json` (with an `order_total` output) plus a full
log + a screenshot to `evidence/discovery-discovery1/`.

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

**4. Assisted fallback (stretch goal)** — replay a copy of the artifact with
one step's locators deliberately corrupted (simulating a vendor UI change
that broke every recorded strategy for that step); without the flag it fails
at that step, with the flag a single bounded, policy-checked LLM call
recovers it:

```bash
python -m src.cli replay --artifact artifacts/add_to_cart_checkout__fault_demo.json \
  --params username=standard_user password=secret_sauce first_name=John last_name=Doe zip_code=94107 \
  --run-id fallback-without-flag   # -> failure at step_3

python -m src.cli replay --artifact artifacts/add_to_cart_checkout__fault_demo.json \
  --params username=standard_user password=secret_sauce first_name=John last_name=Doe zip_code=94107 \
  --allow-assisted-fallback --run-id fallback-with-flag   # -> success, recovers step_3
```

**5. Confidence & approval gating (stretch goal)** — a fresh artifact starts
`"status": "draft"`; measure how consistently it replays, then approve it:

```bash
python -m src.cli stability --artifact artifacts/add_to_cart_checkout.json \
  --params username=standard_user password=secret_sauce first_name=John last_name=Doe zip_code=94107 \
  --runs 5   # writes stability_score back onto the artifact

python -m src.cli approve --artifact artifacts/add_to_cart_checkout.json   # refuses below --min-stability (default 0.8)
```

**6. Stretch goal — agent-facing capability API**, gated on the approval above
(the API is the one genuinely *unattended* caller in this project — every
`/invoke` requires `status: approved`, regardless of the `replay` CLI's own
default):

```bash
python -m src.cli serve
curl http://localhost:8000/capabilities   # shows status + stability_score per capability

# the still-draft fault-injection artifact is refused, no browser launched:
curl -X POST http://localhost:8000/capabilities/add_to_cart_checkout__fault_demo/invoke \
  -H "Content-Type: application/json" \
  -d '{"username":"standard_user","password":"secret_sauce","first_name":"John","last_name":"Doe","zip_code":"94107"}'

# the approved capability succeeds:
curl -X POST http://localhost:8000/capabilities/add_to_cart_checkout/invoke \
  -H "Content-Type: application/json" \
  -d '{"username":"standard_user","password":"secret_sauce","first_name":"John","last_name":"Doe","zip_code":"94107"}'
```

**7. Vision fallback** — the DOM-based path is primary, but when perception
finds zero interactive elements (a canvas-rendered control with no
accessible DOM at all — see `fixtures/canvas_button.html`), discovery falls
back to a screenshot handed to a vision+tool-calling model, which replies
with pixel coordinates. The resulting artifact carries a `coordinates`
locator and replays deterministically like any other:

```bash
python -m src.cli discover \
  --goal "Click the button drawn on the canvas." \
  --target-url "file://$(pwd)/fixtures/canvas_button.html" \
  --app-id canvas-demo --capability-id click_canvas_button \
  --checkpoint-kind text_present --checkpoint-text "Clicked" \
  --viewport 400x200 --headless --run-id vision1

python -m src.cli replay --artifact artifacts/click_canvas_button.json --run-id vision-replay1
```

## Portability check

The pipeline is not saucedemo-specific. The same code, pointed at two more
public sites never seen during development (only `config/allowlist.yaml` +
`config/error_signatures.yaml` extended, as REPORT.md §4 says a real
cross-tenant deployment would), produced genuine artifacts and evidence —
see REPORT.md §4 for the four real bugs this surfaced and fixed:

```bash
python -m src.cli discover \
  --goal "Log in with the given username and password and confirm you reached the secure area." \
  --target-url https://the-internet.herokuapp.com/login \
  --params username=tomsmith password="SuperSecretPassword!" --sensitive password \
  --app-id the-internet-login --capability-id herokuapp_login \
  --checkpoint-kind text_present --checkpoint-text "You logged into a secure area" \
  --headless --run-id portability-heroku1

python -m src.cli discover \
  --goal "Add this product to the cart, view the cart, then click 'Proceed To Checkout' exactly once. As soon as you see a 'Register / Login' link and a 'Continue On Cart' button appear, the goal is complete -- call finish immediately. Do not click 'Proceed To Checkout' more than once, and do not click 'Register / Login'." \
  --target-url https://www.automationexercise.com/product_details/1 \
  --app-id automationexercise --capability-id automationexercise_add_to_cart \
  --checkpoint-kind text_present --checkpoint-text "Register / Login account to proceed on checkout" \
  --headless --run-id portability-ae1

python -m src.cli discover \
  --goal "This product listing page shows products in this fixed reading order: 1) Blue Top, 2) Men Tshirt, 3) Sleeveless Dress, 4) Stylish Dress, 5) Winter Top, 6) Summer White Top, each with its own 'Add to Cart' control in that same order (do not use search or scroll). Click the 3rd 'Add to Cart' element (Sleeveless Dresss) -- do not click the 1st, 2nd, 4th, 5th, or 6th. That click opens an 'Added!' confirmation dialog -- the instant it appears, call finish immediately. Do not click View Cart, do not click Continue Shopping, do not click anything else. Only two actions total: the one click, then finish." \
  --target-url https://www.automationexercise.com/products \
  --app-id automationexercise --capability-id automationexercise_add_to_cart_from_listing \
  --checkpoint-kind text_present --checkpoint-text "Your product has been added to cart." \
  --headless --run-id portability-ae2
```

The third run targets the *listing* page rather than a single product page —
~34 structurally-identical "Add to Cart" buttons instead of one — which is
what surfaced bug #4 (silent ambiguous-locator matches) in REPORT.md §4.

## Running without live services

`pytest tests/` runs fully offline (Chromium + `page.set_content`, no network, no
LLM). `replay` needs network access to the target site but never calls Groq
unless you pass `--allow-assisted-fallback` (only exercised on a step that
already failed). `discover` needs both network and `GROQ_API_KEY`.

## Project layout

```
src/agent/        perception (a11y-tree snapshot, vision fallback) + Groq tool-calling loop
src/artifact/      capability schema, recorder (transcript -> artifact), store
src/replay/        deterministic executor, ranked locator resolution, error taxonomy,
                   bounded assisted-fallback recovery (stretch goal)
src/safety/        allowlist, risk classification, redaction
src/escalation/    human-in-the-loop pause / live handoff / resume
src/evidence/      structured JSONL logging + screenshots
src/capabilities/  stretch goal: artifacts as a callable capability API, gated on approval
config/            allowlist.yaml, error_signatures.yaml
fixtures/          canvas_button.html -- a genuinely DOM-less demo target for the vision fallback
artifacts/         saved capability artifacts
evidence/          discovery + replay run evidence (checked in)
```
