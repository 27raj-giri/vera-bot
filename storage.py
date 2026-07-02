from datetime import datetime, timezone

# -----------------------------
# In-memory Storage
# -----------------------------

contexts = {}

conversations = {}

suppressed = set()


# -----------------------------
# Context Helpers
# -----------------------------

def store_key(scope: str, context_id: str) -> str:
    return f"{scope}:{context_id}"


def get_context(scope: str, context_id: str):
    return contexts.get(store_key(scope, context_id))


def save_context(
    scope: str,
    context_id: str,
    version: int,
    payload: dict,
):

    stored_at = datetime.now(timezone.utc).isoformat()

    contexts[store_key(scope, context_id)] = {
        "version": version,
        "payload": payload,
        "stored_at": stored_at,
    }

    return stored_at


def count_contexts(scope: str):

    return sum(
        1
        for key in contexts
        if key.startswith(f"{scope}:")
    )


# -----------------------------
# Conversation Helpers
# -----------------------------

def get_conversation(conversation_id: str):

    return conversations.setdefault(
        conversation_id,
        [],
    )


def append_turn(
    conversation_id: str,
    role: str,
    body: str,
    **extra,
):

    history = get_conversation(conversation_id)

    history.append(
        {
            "role": role,
            "body": body,
            **extra,
        }
    )


# -----------------------------
# Suppression Helpers
# -----------------------------

def already_sent(
    suppression_key: str,
):

    return suppression_key in suppressed


def mark_sent(
    suppression_key: str,
):

    suppressed.add(suppression_key)