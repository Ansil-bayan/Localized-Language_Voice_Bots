"""
Shared configuration for the Voice AI Call Simulation Engine and the
ASR/LLM/TTS Verification Pipeline.

Both scripts import from here so the LLM model and the voice actors used
for each market/role stay identical across simulation and verification.
"""

import os

# ============================================================================
# LLM CONFIG (shared)
# ============================================================================
# Single Groq-hosted model used by BOTH the call simulation (agent + customer
# turns) and the ASR verification pipeline (reply generation).
#
# NOTE: llama-3.3-70b-versatile was deprecated and shut down by Groq on
# 08/16/26. This is Groq's recommended replacement — check
# https://console.groq.com/docs/deprecations if this ever 404s again, since
# Groq retires models on a rolling basis.
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
# NOTE: .strip() matters here — if the GROQ_API_KEY environment variable was
# set with a trailing newline or stray whitespace (easy to do via copy-paste
# into setx/.env), it gets baked into the "Authorization: Bearer <key>\n"
# HTTP header, which is invalid and makes every request fail. Without
# stripping, that shows up as a confusing httpcore.LocalProtocolError /
# groq.APIConnectionError ("Connection error.") that looks like a network
# problem and retries pointlessly, since the malformed key never changes
# between attempts.
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set (or is empty). Set it as an environment "
        "variable, or in a .env file, before running either script."
    )

# openai/gpt-oss-120b is a REASONING model: it spends part of its token
# budget on internal "thinking" before writing the final reply. With a low
# max_tokens (e.g. 150) and no reasoning_effort set, the reasoning step can
# consume the entire budget and leave literally 0 tokens for the actual
# answer — which shows up as empty "" responses, not an error. Keep
# reasoning effort low and leave enough headroom for a real reply.
GROQ_REASONING_EFFORT = "low"   # 'low' | 'medium' | 'high' for GPT-OSS models
GROQ_MAX_TOKENS = 400            # room for reasoning + a short spoken reply

# ============================================================================
# VOICE ACTOR CONFIG (shared)
# ============================================================================
# One consistent Edge-TTS neural voice per market per role. Using the same
# map in both scripts means the "agent" voice heard in the simulated call
# is the same voice used to render replies in the verification pipeline,
# and the "customer" voice stays consistent wherever customer audio is
# synthesized.
VOICE_MAP = {
    "PH": {
        "agent": "fil-PH-BlessicaNeural",   # Bea
        "customer": "fil-PH-AngeloNeural",
    },
    "ID": {
        "agent": "id-ID-ArdiNeural",        # Mas Budi
        "customer": "id-ID-GadisNeural",
    },
}


def get_voice(market: str, role: str) -> str:
    """Look up the shared voice for a given market ('PH'/'ID') and role ('agent'/'customer')."""
    return VOICE_MAP.get(market, {}).get(role)