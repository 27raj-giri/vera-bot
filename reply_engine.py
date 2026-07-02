from llm import call_groq
from prompt_builder import build_reply_prompt
from json_parser import extract_json

AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "auto reply",
    "auto-reply",
    "out of office",
    "we will respond shortly",
    "our team will get back",
]

STOP_PHRASES = [
    "stop",
    "unsubscribe",
    "leave me alone",
    "not interested",
    "don't message",
    "dont message",
    "remove me",
]

BOOKING_WORDS = [
    "book","booking","appointment","schedule","slot","tomorrow","today",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday","am","pm"
]

def is_auto_reply(msg):
    msg=(msg or "").lower()
    return any(x in msg for x in AUTO_REPLY_PHRASES)

def is_stop(msg):
    msg=(msg or "").lower()
    return any(x in msg for x in STOP_PHRASES)

def is_booking(msg):
    msg=(msg or "").lower()
    return any(x in msg for x in BOOKING_WORDS)

async def process_reply(
    conversation,
    from_role,
    merchant_payload,
    customer_payload,
    incoming_message,
):
    if is_stop(incoming_message):
        return {
            "action":"end",
            "rationale":"User opted out."
        }

    if is_auto_reply(incoming_message):
        count=sum(1 for t in conversation if t.get("auto_reply"))
        if count==0:
            return {
                "action":"send",
                "body":"Looks like an automatic reply. Just reply 'Yes' whenever you're available and we'll continue.",
                "cta":"binary_yes_no",
                "rationale":"First auto reply."
            }
        if count==1:
            return {
                "action":"wait",
                "wait_seconds":86400,
                "rationale":"Second auto reply."
            }
        return {
            "action":"end",
            "rationale":"Repeated auto replies."
        }

    if from_role=="customer" and is_booking(incoming_message):
        return {
            "action":"send",
            "body":"Thank you! Your requested slot has been noted. We'll confirm the appointment shortly.",
            "cta":"none",
            "rationale":"Booking intent detected."
        }

    system_prompt,user_prompt=build_reply_prompt(
        conversation,
        from_role,
        merchant_payload,
        customer_payload,
        incoming_message,
    )

    try:
        raw=await call_groq(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=500,
        )
        result=extract_json(raw)

        if result.get("action") not in {"send","wait","end"}:
            result["action"]="send"

        if result["action"]=="send":
            result.setdefault("cta","open_ended")
            if not result.get("body"):
                raise ValueError("Empty body")

        return result

    except Exception:
        if from_role=="customer":
            return {
                "action":"send",
                "body":"Thanks! We've received your request and will get back to you shortly.",
                "cta":"none",
                "rationale":"Customer fallback."
            }

        return {
            "action":"send",
            "body":"Thanks for the update. I'll prepare the best next recommendation for your business.",
            "cta":"open_ended",
            "rationale":"Merchant fallback."
        }
