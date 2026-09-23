import json
import os
import sys
import time

from groq import Groq, RateLimitError, APIStatusError, APIConnectionError
from gtts import gTTS

from shared_config import GROQ_MODEL_NAME, GROQ_API_KEY, GROQ_REASONING_EFFORT, GROQ_MAX_TOKENS, get_voice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Single Groq client, shared model — same model used in the ASR verification pipeline
groq_client = Groq(api_key=GROQ_API_KEY)


def load_agent_config(filename: str) -> dict:
    """Loads and prepares agent configuration from a JSON specification file."""
    filepath = os.path.join(BASE_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Convenience accessors for backwards compatibility
    if "name" not in config:
        config["name"] = config.get("persona", {}).get("name", "Agent")
    if "rules" not in config:
        config["rules"] = config.get("system_instructions", "")

    return config


# System Rules & Persona Configurations loaded from JSON specifications
PH_AGENT_CONFIG = load_agent_config("ph_agent.json")
ID_AGENT_CONFIG = load_agent_config("id_agent.json")

# The 6 Assignment Test Scenarios (PH 1-3 & ID 1-3)
SCENARIOS = [
    {
        "id": "PH_Call_01_Cooperative",
        "market": "PH",
        "tts_lang": "tl",
        "agent_config": PH_AGENT_CONFIG,
        "customer_prompt": "You are Marco Cruz, a cooperative bank customer in Manila. You are busy but polite. When Bea calls about life insurance/bancassurance, ask about the monthly premium in Taglish. When she explains it's 2,000 Pesos/month, agree to meet a financial advisor.",
        "max_turns": 4,
    },
    {
        "id": "PH_Call_02_Objection_Escalation",
        "market": "PH",
        "tts_lang": "tl",
        "agent_config": PH_AGENT_CONFIG,
        "customer_prompt": "You are Sheila, an upset customer calling about your insurance policy. Dispute a sudden increase in your premium due to a rider charge. Demand to be transferred to a human agent immediately.",
        "max_turns": 3,
    },
    {
        "id": "PH_Call_03_Grace_Period_Lapse",
        "market": "PH",
        "tts_lang": "tl",
        "agent_config": PH_AGENT_CONFIG,
        "customer_prompt": "You are Mrs. Elena Ramos, a long-time bank customer in Quezon City. You are concerned because your auto-debit for your life insurance premium failed due to an expired debit card. You are worried your policy might lapse and lose coverage benefits. In Taglish, explain your card issue, ask how much time you have left in the grace period, and agree to settle the premium via online banking today.",
        "max_turns": 4,
    },
    {
        "id": "ID_Call_01_Cooperative",
        "market": "ID",
        "tts_lang": "id",
        "agent_config": ID_AGENT_CONFIG,
        "customer_prompt": "You are Mas Rizky, a friendly customer in Jakarta. Mas Budi is calling to remind you about your motorcycle loan installment (angsuran) of Rp 850,000 due in 2 days. Agree to pay via BCA Virtual Account.",
        "max_turns": 4,
    },
    {
        "id": "ID_Call_02_Javanese_Dispute",
        "market": "ID",
        "tts_lang": "id",
        "agent_config": ID_AGENT_CONFIG,
        "customer_prompt": "You are Pak Bambang, a farmer from Central Java with a heavy Javanese speech accent (use Javanese-influenced Bahasa Indonesia like 'iyo', 'yo', 'wis', 'sampeyan'). Dispute a late fee (denda) of Rp 2,150,000 because your harvest was delayed. Demand to speak to a supervisor.",
        "max_turns": 4,
    },
    {
        "id": "ID_Call_03_Hardship_Janji_Bayar",
        "market": "ID",
        "tts_lang": "id",
        "agent_config": ID_AGENT_CONFIG,
        "customer_prompt": "You are Ibu Nurul, a small catering business owner in Bandung. Mas Budi is calling regarding your monthly car financing installment (angsuran) of Rp 3,200,000 which is 5 days past due. Politely explain that your clients have delayed payments to you, ask if there is a penalty (denda), and commit to paying via Virtual Account this Friday (janji bayar).",
        "max_turns": 4,
    },
]


def generate_llm_response(system_prompt, conversation_history, max_retries: int = 4):
    """Generates dialogue turns using the shared Groq-hosted model.

    Retries with exponential backoff on rate limits / transient errors so a
    single API hiccup doesn't take down the whole multi-scenario run. Raises
    after exhausting retries so the caller can decide how to handle it
    (e.g. skip this scenario and continue with the next one).
    """
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    last_error = None
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=messages,
                temperature=0.6,
                max_tokens=GROQ_MAX_TOKENS,
                reasoning_effort=GROQ_REASONING_EFFORT,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            # gpt-oss-120b spent its whole token budget on internal reasoning
            # and left nothing for the actual reply. Treat this as retryable
            # rather than silently returning "".
            last_error = "model returned an empty reply (reasoning likely consumed the token budget)"
            print(f"[!] Empty reply from Groq, retrying (attempt {attempt + 1}/{max_retries})...")
        except RateLimitError as e:
            last_error = e
            wait = 2 ** attempt  # 1, 2, 4, 8 seconds
            print(f"[!] Groq rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
        except (APIStatusError, APIConnectionError) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"[!] Groq API error ({e}), retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError(f"Groq call failed after {max_retries} attempts: {last_error}")


def save_audio_gtts(text, lang, filename):
    """Converts the full call script into an .mp3 audio file using gTTS."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)


def render_call_audio(text: str, market: str, lang: str, filename: str, voice_role: str = "agent"):
    """Synthesizes dialogue using the shared Edge-TTS neural voice for this market/role,
    falling back to gTTS. voice_role is 'agent' or 'customer' and is looked up from
    shared_config.VOICE_MAP so the same voice actor is used across both scripts.
    """
    voice = get_voice(market, voice_role)
    if voice:
        try:
            import asyncio
            import edge_tts

            async def _synth():
                comm = edge_tts.Communicate(text, voice)
                await comm.save(filename)

            asyncio.run(_synth())
            print(f"✓ Generated high-fidelity Edge-TTS neural recording ({voice}): {filename}")
            return
        except Exception as e:
            print(f"[*] Edge-TTS fallback ({e}), falling back to gTTS.")

    save_audio_gtts(text, lang, filename)
    print(f"✓ Generated audio recording (gTTS): {filename}")


def run_scenario(scenario: dict, evaluate_with_asr: bool):
    """Runs a single scenario end-to-end: dialogue generation, transcript save,
    audio rendering, and (optionally) ASR verification. Raises on unrecoverable
    errors so the caller can log and continue with the next scenario."""
    sc_id = scenario["id"]
    market = scenario["market"]
    agent_cfg = scenario["agent_config"]

    print(f"\n--- RUNNING SCENARIO: {sc_id} [{market}] ---")

    fallback_rule = (
        "CRITICAL FALLBACK RULE: Under no circumstances switch into English when confused or responding to customer complaints. "
        "You must remain in the customer's native conversational register (Taglish with 'po'/'ho' for PH, polite Bahasa Indonesia with 'Pak/Bu' for ID)."
    )

    if "system_instructions" in agent_cfg:
        agent_system_prompt = (
            f"{agent_cfg['system_instructions']} {fallback_rule} "
            f"Keep responses brief (1-3 sentences max), conversational and ready for speech output."
        )
    else:
        persona_desc = (
            f"{agent_cfg['persona'].get('role', '')} ({agent_cfg['persona'].get('institution', '')})"
            if isinstance(agent_cfg.get("persona"), dict)
            else agent_cfg.get("persona", "")
        )
        agent_system_prompt = (
            f"You are {agent_cfg['name']}, {persona_desc}. "
            f"Rules: {agent_cfg.get('rules', '')}. {fallback_rule} "
            f"Keep responses brief (1-3 sentences max), conversational and ready for speech output."
        )
    customer_system_prompt = (
        f"{scenario['customer_prompt']} "
        f"Keep responses short and realistic for a phone call (1-2 sentences)."
    )

    agent_history = []
    customer_history = []
    full_transcript = []
    customer_lines = []  # customer-only lines, kept separate from the mixed transcript

    # Turn 1: Agent Call Opening
    if sc_id == "PH_Call_01_Cooperative":
        agent_first_turn = "Good day po! Am I speaking with Mr. Marco Cruz?"
    elif sc_id == "PH_Call_02_Objection_Escalation":
        agent_first_turn = "Hello po, Ma'am Sheila! Good morning. Si Bea po ito galing sa inyong bank partner."
    elif sc_id == "PH_Call_03_Grace_Period_Lapse":
        agent_first_turn = "Magandang araw po, Mrs. Elena Ramos! Si Bea po ito mula sa Bancassurance Division ng inyong bank branch. Kumusta po kayo?"
    elif sc_id == "ID_Call_01_Cooperative":
        agent_first_turn = "Selamat pagi, bisa bicara dengan Mas Rizky?"
    elif sc_id == "ID_Call_02_Javanese_Dispute":
        agent_first_turn = "Selamat siang Bapak Bambang, betul nggih ini dengan Bapak sendiri?"
    elif sc_id == "ID_Call_03_Hardship_Janji_Bayar":
        agent_first_turn = "Selamat siang Ibu Nurul, perkenalkan saya Mas Budi dari IndoFinance. Semoga Ibu sekeluarga selalu sehat ya, Bu."
    else:
        agent_first_turn = "Hello, good day!"

    print(f"[{agent_cfg['name']} (AI)]: {agent_first_turn}")
    full_transcript.append(f"{agent_cfg['name']} (AI): {agent_first_turn}")

    agent_history.append({"role": "assistant", "content": agent_first_turn})
    customer_history.append({"role": "user", "content": agent_first_turn})

    for turn in range(scenario["max_turns"]):
        # 1. Customer Responds
        customer_reply = generate_llm_response(customer_system_prompt, customer_history)
        print(f"[Customer]: {customer_reply}")
        full_transcript.append(f"Customer: {customer_reply}")
        customer_lines.append(customer_reply)

        customer_history.append({"role": "assistant", "content": customer_reply})
        agent_history.append({"role": "user", "content": customer_reply})

        # Check for Escalation Triggers
        customer_lower = customer_reply.lower()
        escalation_cfg = agent_cfg.get("conversation_flows", {}).get("escalation_trigger", {})
        escalation_keywords = escalation_cfg.get("keywords", ["human agent", "supervisor", "kuasane", "totoong tao", "pimpinan"])

        if any(term in customer_lower for term in escalation_keywords):
            default_escalation = (
                "Ico-connect ko na po kayo sa ating Senior Financial Specialist. Hold the line po."
                if market == "PH"
                else "Baik, Budi bantu sambungkan ke Supervisor Restrukturisasi kami ya Pak/Bu."
            )
            escalation_msg = escalation_cfg.get("response_script", default_escalation).replace("{honorific}", "Pak/Bu")
            print(f"[{agent_cfg['name']} (AI)]: {escalation_msg}")
            print("[SYSTEM EVENT]: SIP_TRANSFER_TRIGGERED -> Transferring to Human Agent")
            full_transcript.append(f"{agent_cfg['name']} (AI): {escalation_msg}")
            full_transcript.append("[SYSTEM EVENT]: SIP_TRANSFER_TRIGGERED -> Transferred to Human Agent")
            break

        # 2. Voice Agent Responds
        agent_reply = generate_llm_response(agent_system_prompt, agent_history)
        print(f"[{agent_cfg['name']} (AI)]: {agent_reply}")
        full_transcript.append(f"{agent_cfg['name']} (AI): {agent_reply}")

        agent_history.append({"role": "assistant", "content": agent_reply})
        customer_history.append({"role": "user", "content": agent_reply})

    # Save Text Transcript
    transcript_filename = f"{sc_id}_transcript.txt"
    with open(transcript_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(full_transcript))
    print(f"✓ Saved transcript: {transcript_filename}")

    # Render full-call speech audio (both speakers) for human review / archival,
    # using the shared "agent" voice for the market.
    audio_filename = f"{sc_id}.mp3"
    spoken_dialogue = " ".join([line for line in full_transcript if not line.startswith("[SYSTEM EVENT]")])
    render_call_audio(spoken_dialogue, market, scenario["tts_lang"], audio_filename, voice_role="agent")

    # Render a CUSTOMER-ONLY audio clip using the shared "customer" voice.
    # This is what a real inbound/outbound ASR system would actually be
    # listening to, so it's the correct input for the verification pipeline.
    customer_audio_filename = f"{sc_id}_customer.mp3"
    customer_only_speech = " ".join(customer_lines)
    if customer_only_speech.strip():
        render_call_audio(
            customer_only_speech,
            market,
            scenario["tts_lang"],
            customer_audio_filename,
            voice_role="customer",
        )
    else:
        print(f"[!] No customer lines captured for {sc_id}; skipping customer-only audio.")

    # Pipeline connection: Verify CUSTOMER-ONLY audio with Faster-Whisper ASR
    if evaluate_with_asr and customer_only_speech.strip():
        evaluate_call_with_asr(customer_audio_filename, market=market, session_id=sc_id)


def run_simulation(evaluate_with_asr: bool = False, market_filter: str = "ALL"):
    print("=" * 65)
    print(f"STARTING VOICE AI CALL SIMULATION (GROQ: {GROQ_MODEL_NAME} + NEURAL TTS)")
    if evaluate_with_asr:
        print("[*] Integrated Faster-Whisper ASR + Groq LLM verification enabled.")
    print("=" * 65)

    target_scenarios = [s for s in SCENARIOS if market_filter in ["ALL", s["market"]]]

    completed = []
    failed = []

    for scenario in target_scenarios:
        sc_id = scenario["id"]
        try:
            run_scenario(scenario, evaluate_with_asr)
            completed.append(sc_id)
        except Exception as e:
            # A failure in one scenario (e.g. Groq retries exhausted) is
            # logged and skipped rather than crashing the whole run, so the
            # other scenarios still get generated.
            print(f"[!] SCENARIO FAILED: {sc_id} -> {e}")
            print(f"[!] Skipping to next scenario...")
            failed.append(sc_id)

    print("\n" + "=" * 65)
    print(f"SIMULATION COMPLETE: {len(completed)}/{len(target_scenarios)} scenarios succeeded.")
    if failed:
        print(f"Failed scenarios: {', '.join(failed)}")
    print("=" * 65)


def evaluate_call_with_asr(audio_path: str, market: str, session_id: str):
    """Feeds customer-only simulation audio into ASR_Engines pipeline for transcription and verification."""
    try:
        from ASR_Engines import run_asr_pipeline
        print(f"\n[PIPELINE CONNECTION] Feeding customer-only audio '{audio_path}' into Faster-Whisper ASR & Groq Pipeline...")
        run_asr_pipeline(audio_path, market=market, session_id=session_id)
    except Exception as e:
        print(f"[!] ASR pipeline evaluation notice: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice AI Call Simulation Engine")
    parser.add_argument("--market", choices=["PH", "ID", "ALL"], default="ALL", help="Target market to simulate (PH, ID, or ALL)")
    parser.add_argument("--asr", action="store_true", help="Pass generated customer-only audio directly to Faster-Whisper ASR pipeline")
    args = parser.parse_args()

    run_simulation(evaluate_with_asr=args.asr, market_filter=args.market)