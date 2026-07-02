import os
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(Exception):
    pass


async def call_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:

    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY not configured")

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:

        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=payload,
        )

    if response.status_code != 200:
        raise LLMError(response.text)

    data = response.json()

    if "choices" not in data:
        raise LLMError(str(data))

    return data["choices"][0]["message"]["content"].strip()