from llm import call_groq
from prompt_builder import build_compose_prompt
from json_parser import extract_json
from storage import append_turn

VALID_CTAS = {
    "binary_yes_no",
    "binary_confirm_cancel",
    "multi_choice_slot",
    "open_ended",
    "none",
}

def _fallback(kind, merchant_name):
    mapping = {
        "competitor_opened": (
            f"{merchant_name}, a nearby competitor has become more active. "
            "Would you like a 2‑minute action plan to protect your customer flow?",
            "binary_yes_no",
        ),
        "festival": (
            f"{merchant_name}, there's an upcoming seasonal opportunity. "
            "Would you like a ready-to-use promotional idea?",
            "binary_yes_no",
        ),
        "performance_dip": (
            f"{merchant_name}, we've noticed a recent performance dip. "
            "Would you like recommendations to recover quickly?",
            "binary_yes_no",
        ),
        "research_digest": (
            f"{merchant_name}, a new industry insight could affect your business. "
            "Would you like a short summary?",
            "binary_yes_no",
        ),
        "regulation_change": (
            f"{merchant_name}, there's an important compliance update. "
            "Would you like the key actions to stay compliant?",
            "binary_yes_no",
        ),
        "recall_due": (
            "A customer may be due for a follow-up. Would you like to send a reminder?",
            "binary_yes_no",
        ),
    }
    return mapping.get(kind, (
        "We noticed an important business update. Would you like a quick summary?",
        "binary_yes_no",
    ))

async def compose_action(
    merchant_payload,
    category_payload,
    trigger_payload,
    customer_payload,
    trigger_id,
    now,
):
    system_prompt, user_prompt = build_compose_prompt(
        merchant_payload,
        category_payload,
        trigger_payload,
        customer_payload,
    )

    kind = trigger_payload.get("kind", "generic")
    merchant_name = merchant_payload.get("identity", {}).get("name", "Merchant")

    try:
        raw = await call_groq(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=700,
        )

        print("\n========== GROQ RAW ==========")
        print(raw)
        print("==============================\n")
        
        result = extract_json(raw)
    except Exception:
        body, cta = _fallback(kind, merchant_name)
        result = {
            "body": body,
            "cta": cta,
            "send_as": "merchant_on_behalf" if customer_payload else "vera",
            "rationale": "Fallback response.",
        }

    if not result.get("body"):
        body, cta = _fallback(kind, merchant_name)
        result["body"] = body
        result["cta"] = cta

    if result.get("cta") not in VALID_CTAS:
        result["cta"] = "open_ended"

    if result.get("send_as") not in {"vera", "merchant_on_behalf"}:
        result["send_as"] = "vera"

    merchant_id = trigger_payload.get("merchant_id")
    customer_id = trigger_payload.get("customer_id")
    suppression_key = trigger_payload.get("suppression_key", trigger_id)
    conversation_id = f"conv_{merchant_id}_{trigger_id}"

    append_turn(
        conversation_id=conversation_id,
        role="vera",
        body=result["body"],
        ts=now,
    )

    return {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": "merchant_on_behalf" if customer_id else result["send_as"],
        "trigger_id": trigger_id,
        "template_name": f"vera_{kind}_v1",
        "template_params": [],
        "body": result["body"],
        "cta": result["cta"],
        "suppression_key": suppression_key,
        "rationale": result.get("rationale", ""),
    }
