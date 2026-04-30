# Vera Bot — Magicpin AI Challenge Submission

**Team:** Aayush Raj Giri  
**Contact:** heyayush27@gmail.com  
**Model:** Groq - Llama 3.3 70B Versatile

## Approach

Single-prompt composer grounded in full merchant + category + trigger context.

**How it works:**
1. `/v1/context` stores all incoming context (category, merchant, customer, trigger) in memory with version control
2. `/v1/tick` picks the highest-urgency available trigger, pulls all relevant context, builds a detailed prompt, and calls Groq LLM to compose a specific, actionable message
3. `/v1/reply` handles merchant replies — detects auto-replies (with graceful backoff), opt-outs, and normal replies — using LLM to continue conversations naturally

## Key Design Decisions

- **One action per tick** — restraint over spam. Quality over volume.
- **Suppression enforced** — no repeat messages for the same suppression key
- **Auto-reply detection** — progressive backoff (send prompt → wait 24h → end)
- **Context grounding** — every message uses real numbers, real offers, real signals from the merchant data
- **Category voice matching** — tone adapts based on category (clinical for dentists, warm for salons, etc.)

## Stack

- FastAPI + Uvicorn (Python)
- Groq API (Llama 3.3 70B) — free tier
- Railway deployment — zero cost
- Pure in-memory state (no database needed for challenge scope)

## Tradeoffs

- In-memory storage resets on restart — acceptable for 3-day challenge window
- Single action per tick is conservative but avoids spam penalties
- Temperature set to 0.3 for deterministic-leaning outputs
