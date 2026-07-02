import json
import re


class JSONParseError(Exception):
    pass


def extract_json(raw: str) -> dict:
    """
    Robust JSON extractor for LLM responses.

    Handles:
    - Plain JSON
    - ```json ... ```
    - Extra explanation before/after JSON
    """

    if not raw:
        raise JSONParseError("Empty response")

    raw = raw.strip()

    # Remove markdown fences
    raw = re.sub(r"^```json", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```", "", raw)
    raw = re.sub(r"```$", "", raw)

    raw = raw.strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Find first JSON object
    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise JSONParseError(
        f"Unable to parse JSON.\n\nResponse:\n{raw}"
    )


def safe_get(data: dict, key: str, default=None):
    if not isinstance(data, dict):
        return default
    return data.get(key, default)