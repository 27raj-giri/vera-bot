from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import time

from storage import (
    contexts,
    conversations,
    suppressed,
    store_key,
    get_context,
    save_context,
    count_contexts,
    already_sent,
    mark_sent,
)

from compose_engine import compose_action
from reply_engine import process_reply

app = FastAPI()
START_TIME = time.time()


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
        },
    }



@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Aayush Raj Giri",
        "team_members": ["Aayush Raj Giri"],
        "model": "llama-3.3-70b-versatile",
        "approach": "Groq LLM composer grounded in merchant + category + trigger context. Stateful in-memory store with idempotent context versioning. Separate compose and reply engines.",
        "contact_email": "heyayush27@gmail.com",
        "version": "2.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/v1/context")
async def push_context(request: Request):
    body = await request.json()

    scope = body["scope"]
    context_id = body["context_id"]
    version = body.get("version", 1)
    payload = body.get("payload", {})

    key = store_key(scope, context_id)
    existing = contexts.get(key)

    if existing:
        current_version = existing["version"]

        if version < current_version:
            return JSONResponse(
                status_code=409,
                content={
                    "accepted": False,
                    "reason": "stale_version",
                    "current_version": current_version,
                },
            )

        if version == current_version:
            return {
                "accepted": True,
                "ack_id": f"ack_{context_id}_v{version}",
                "stored_at": existing["stored_at"],
                "idempotent": True,
            }

    stored_at = save_context(
        scope,
        context_id,
        version,
        payload,
    )

    return {
        "accepted": True,
        "ack_id": f"ack_{context_id}_v{version}",
        "stored_at": stored_at,
    }


@app.post("/v1/tick")
async def tick(request: Request):
    body = await request.json()

    actions = []

    for trigger_id in body.get("available_triggers", []):
        
        trigger_ctx = get_context("trigger", trigger_id)
        if not trigger_ctx:
            continue

        trigger_payload = trigger_ctx["payload"]

        suppression_key = trigger_payload.get(
            "suppression_key",
            trigger_id,
        )

        if already_sent(suppression_key):
            continue

        merchant_ctx = get_context(
            "merchant",
            trigger_payload["merchant_id"],
        )

        if not merchant_ctx:
            continue

        merchant_payload = merchant_ctx["payload"]

        category_slug = merchant_payload.get(
            "category_slug",
            "",
        )

        category_ctx = get_context(
            "category",
            category_slug,
        )

        category_payload = (
            category_ctx["payload"]
            if category_ctx
            else {}
        )

        customer_payload = None

        customer_id = trigger_payload.get("customer_id")

        if customer_id:
            ctx = get_context(
                "customer",
                customer_id,
            )
            if ctx:
                customer_payload = ctx["payload"]

        action = await compose_action(
            merchant_payload,
            category_payload,
            trigger_payload,
            customer_payload,
            trigger_id,
            datetime.now(timezone.utc).isoformat(),
        )

        mark_sent(suppression_key)

        actions.append(action)

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(request: Request):
    body = await request.json()

    conversation_id = body["conversation_id"]

    history = conversations.setdefault(
        conversation_id,
        [],
    )

    result = await process_reply(
        conversation=history,
        from_role=body.get("from_role", "merchant"),
        merchant_payload=(
            get_context(
                "merchant",
                body["merchant_id"],
            )["payload"]
            if get_context("merchant", body["merchant_id"])
            else {}
        ),
        customer_payload=(
            get_context(
                "customer",
                body["customer_id"],
            )["payload"]
            if body.get("customer_id")
            and get_context(
                "customer",
                body["customer_id"],
            )
            else None
        ),
        incoming_message=body.get("message", ""),
    )

    return result
