from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import json
import os
import time
from datetime import datetime, timezone

app = FastAPI()

# ── In-memory storage ──────────────────────────────────────────────────────────
contexts = {}          # key: f"{scope}:{context_id}"  value: {version, payload}
conversations = {}     # key: conversation_id           value: list of turns
suppressed = set()     # suppression_keys already sent

START_TIME = time.time()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


# ── Helpers ────────────────────────────────────────────────────────────────────

def store_key(scope: str, context_id: str) -> str:
    return f"{scope}:{context_id}"


def get_context(scope: str, context_id: str):
    return contexts.get(store_key(scope, context_id))


def count_contexts(scope: str) -> int:
    return sum(1 for k in contexts if k.startswith(f"{scope}:"))


async def call_groq(system_prompt: str, user_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }
    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(GROQ_URL, headers=headers, json=body)
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def build_compose_prompt(merchant_payload: dict, category_payload: dict, trigger_payload: dict, customer_payload: dict | None) -> tuple[str, str]:
    system = """You are Vera, magicpin's AI assistant for merchant growth in India.

Your job: compose ONE highly specific, context-aware message for a merchant.

RULES:
- Use real numbers, dates, and offers from the context
- Keep the message under 200 words
- One clear CTA only
- No URLs in message body
- Match the category tone (dentist = clinical peer, salon = warm visual, restaurant = timely utility, gym = motivational, pharmacy = clinical trustworthy)
- Write in the merchant's language preference (Hindi/English mix if needed)
- Be specific about WHY NOW matters

Return ONLY valid JSON in this exact format:
{
  "body": "the message text",
  "cta": "binary_yes_no OR open_ended OR multi_choice_slot OR binary_confirm_cancel OR none",
  "send_as": "vera OR merchant_on_behalf",
  "rationale": "one sentence explaining the decision"
}"""

    merchant_name = merchant_payload.get("identity", {}).get("name", "Merchant")
    owner_name = merchant_payload.get("identity", {}).get("owner_first_name", "")
    city = merchant_payload.get("identity", {}).get("locality", "")
    category = merchant_payload.get("category_slug", "")
    offers = merchant_payload.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    performance = merchant_payload.get("performance", {})
    signals = merchant_payload.get("signals", [])
    conversation_history = merchant_payload.get("conversation_history", [])
    reviews = merchant_payload.get("review_themes", [])
    customer_agg = merchant_payload.get("customer_aggregate", {})

    trigger_kind = trigger_payload.get("kind", "")
    trigger_urgency = trigger_payload.get("urgency", 1)
    trigger_data = trigger_payload.get("payload", {})

    category_voice = category_payload.get("voice", {}) if category_payload else {}
    category_digest = category_payload.get("digest", []) if category_payload else []

    customer_info = ""
    if customer_payload:
        customer_info = f"""
CUSTOMER CONTEXT:
{json.dumps(customer_payload, indent=2)}
"""

    user = f"""MERCHANT: {merchant_name} ({category}) in {city}
OWNER: {owner_name}

PERFORMANCE (last 30 days):
- Views: {performance.get('views', 'N/A')}
- Calls: {performance.get('calls', 'N/A')}
- CTR: {performance.get('ctr', 'N/A')}
- 7-day delta: views {performance.get('delta_7d', {}).get('views_pct', 0)*100:.0f}%, calls {performance.get('delta_7d', {}).get('calls_pct', 0)*100:.0f}%

ACTIVE OFFERS:
{json.dumps(active_offers, indent=2) if active_offers else 'None'}

SIGNALS: {', '.join(signals)}

RECENT REVIEWS: {json.dumps(reviews, indent=2) if reviews else 'None'}

CUSTOMER AGGREGATE: {json.dumps(customer_agg, indent=2)}

RECENT CONVERSATION:
{json.dumps(conversation_history[-3:], indent=2) if conversation_history else 'No prior conversation'}

TRIGGER (WHY NOW):
Kind: {trigger_kind}
Urgency: {trigger_urgency}/5
Data: {json.dumps(trigger_data, indent=2)}

CATEGORY VOICE: {json.dumps(category_voice, indent=2)}
CATEGORY DIGEST: {json.dumps(category_digest, indent=2) if category_digest else 'None'}
{customer_info}

Compose the best possible Vera message for this exact situation. Return only JSON."""

    return system, user


def build_reply_prompt(conversation_id: str, merchant_reply: str, conversation_history: list, merchant_payload: dict) -> tuple[str, str]:
    system = """You are Vera, magicpin's AI assistant for merchant growth.

A merchant just replied to your message. Decide what to do next.

RULES:
- If reply is an auto-reply (canned "Thank you for contacting..."), respond with action=wait on first occurrence, then end after repeated auto-replies
- If merchant says stop/not interested/go away, respond with action=end
- If merchant asks something out of scope, politely decline and redirect
- If merchant is engaged, continue the conversation helpfully with one clear CTA
- No URLs in body
- Keep response under 150 words

Return ONLY valid JSON:
{
  "action": "send OR wait OR end",
  "body": "message text (only if action=send)",
  "cta": "binary_yes_no OR open_ended OR binary_confirm_cancel OR none (only if action=send)",
  "wait_seconds": 14400 (only if action=wait),
  "rationale": "one sentence"
}"""

    owner_name = merchant_payload.get("identity", {}).get("owner_first_name", "") if merchant_payload else ""

    user = f"""CONVERSATION ID: {conversation_id}
MERCHANT OWNER: {owner_name}

CONVERSATION HISTORY (last 5 turns):
{json.dumps(conversation_history[-5:], indent=2)}

MERCHANT JUST REPLIED: "{merchant_reply}"

What should Vera do next? Return only JSON."""

    return system, user


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": {
            "category": count_contexts("category"),
            "merchant": count_contexts("merchant"),
            "customer": count_contexts("customer"),
            "trigger": count_contexts("trigger"),
        }
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Aayush Raj Giri",
        "team_members": ["Aayush Raj Giri"],
        "model": MODEL,
        "approach": "Groq LLM composer grounded in merchant + category + trigger context. Stateful in-memory store with idempotent version control.",
        "contact_email": "heyayush27@gmail.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(request: Request):
    body = await request.json()
    scope = body.get("scope")
    context_id = body.get("context_id")
    version = body.get("version", 1)
    payload = body.get("payload", {})
    delivered_at = body.get("delivered_at", datetime.now(timezone.utc).isoformat())

    key = store_key(scope, context_id)
    existing = contexts.get(key)

    if existing:
        if existing["version"] == version:
            return JSONResponse(
                status_code=409,
                content={"accepted": False, "reason": "stale_version", "current_version": existing["version"]}
            )

    stored_at = datetime.now(timezone.utc).isoformat()
    contexts[key] = {"version": version, "payload": payload, "stored_at": stored_at}

    ack_id = f"ack_{context_id}_v{version}".replace(" ", "_")
    return {"accepted": True, "ack_id": ack_id, "stored_at": stored_at}


@app.post("/v1/tick")
async def tick(request: Request):
    body = await request.json()
    available_triggers = body.get("available_triggers", [])
    now = body.get("now", datetime.now(timezone.utc).isoformat())

    if not available_triggers:
        return {"actions": []}

    actions = []

    for trigger_id in available_triggers:
        trigger_ctx = get_context("trigger", trigger_id)
        if not trigger_ctx:
            continue

        trigger_payload = trigger_ctx["payload"]
        suppression_key = trigger_payload.get("suppression_key", trigger_id)

        if suppression_key in suppressed:
            continue

        merchant_id = trigger_payload.get("merchant_id")
        customer_id = trigger_payload.get("customer_id")

        if not merchant_id:
            continue

        merchant_ctx = get_context("merchant", merchant_id)
        if not merchant_ctx:
            continue

        merchant_payload = merchant_ctx["payload"]
        category_slug = merchant_payload.get("category_slug", "")
        category_ctx = get_context("category", category_slug)
        category_payload = category_ctx["payload"] if category_ctx else {}

        customer_payload = None
        if customer_id:
            customer_ctx = get_context("customer", customer_id)
            if customer_ctx:
                customer_payload = customer_ctx["payload"]

        try:
            system_prompt, user_prompt = build_compose_prompt(
                merchant_payload, category_payload, trigger_payload, customer_payload
            )
            raw = await call_groq(system_prompt, user_prompt)

            # strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
        except Exception as e:
            print(f"ERROR in tick: {e}")
            continue

        conversation_id = f"conv_{merchant_id}_{trigger_id}"
        send_as = result.get("send_as", "vera")

        if customer_id:
            send_as = "merchant_on_behalf"

        action = {
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger_payload.get('kind', 'generic')}_v1",
            "template_params": [],
            "body": result.get("body", ""),
            "cta": result.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": result.get("rationale", ""),
        }

        conversations[conversation_id] = [
            {"role": "vera", "body": result.get("body", ""), "ts": now}
        ]

        suppressed.add(suppression_key)
        actions.append(action)

        # Only send one action per tick to avoid spam penalty
        break

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(request: Request):
    body = await request.json()
    conversation_id = body.get("conversation_id", "")
    merchant_id = body.get("merchant_id", "")
    customer_id = body.get("customer_id")
    merchant_message = body.get("message", "")
    turn_number = body.get("turn_number", 1)

    # detect auto-reply
    auto_reply_phrases = [
        "thank you for contacting",
        "we will respond shortly",
        "our team will get back",
        "auto-reply",
        "out of office",
    ]
    is_auto_reply = any(p in merchant_message.lower() for p in auto_reply_phrases)

    # count auto-replies in this conversation
    conv_history = conversations.get(conversation_id, [])
    auto_reply_count = sum(1 for t in conv_history if t.get("is_auto_reply"))

    if is_auto_reply:
        conv_history.append({"role": "merchant", "body": merchant_message, "is_auto_reply": True})
        conversations[conversation_id] = conv_history

        if auto_reply_count == 0:
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' to continue.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply on first turn; one explicit prompt to flag it for the owner."
            }
        elif auto_reply_count == 1:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Auto-reply twice in a row; owner not available. Waiting 24 hours before retry."
            }
        else:
            return {
                "action": "end",
                "rationale": "Auto-reply 3 or more times in a row. No real engagement; closing conversation."
            }

    # detect opt-out
    opt_out_phrases = [
        "stop", "not interested", "dont message", "don't message",
        "stop messaging", "go away", "useless", "bothering me", "leave me alone"
    ]
    is_opt_out = any(p in merchant_message.lower() for p in opt_out_phrases)

    if is_opt_out:
        conv_history.append({"role": "merchant", "body": merchant_message})
        conversations[conversation_id] = conv_history
        return {
            "action": "send",
            "body": "Apologies — I won't message again. If anything changes, just reply 'Hi Vera' to restart. 🙏",
            "cta": "none",
            "rationale": "Merchant opted out explicitly; sending one graceful closing message then ending."
        }

    # normal reply — use Groq to respond
    conv_history.append({"role": "merchant", "body": merchant_message})
    conversations[conversation_id] = conv_history

    merchant_ctx = get_context("merchant", merchant_id)
    merchant_payload = merchant_ctx["payload"] if merchant_ctx else {}

    try:
        system_prompt, user_prompt = build_reply_prompt(
            conversation_id, merchant_message, conv_history, merchant_payload
        )
        raw = await call_groq(system_prompt, user_prompt)

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
    except Exception as e:
        return {
            "action": "send",
            "body": "Got it! Let me help you with that. What would you like to do next?",
            "cta": "open_ended",
            "rationale": "Fallback response due to processing error."
        }

    if result.get("action") == "send":
        conv_history.append({"role": "vera", "body": result.get("body", "")})
        conversations[conversation_id] = conv_history

    return result
