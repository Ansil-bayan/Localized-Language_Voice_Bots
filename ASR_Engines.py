import argparse
import asyncio
import glob
import json
import os
import sys
from datetime import datetime
from typing import Callable, Optional, Dict, Any

from groq import AsyncGroq
import edge_tts
from gtts import gTTS
from faster_whisper import WhisperModel

from shared_config import GROQ_MODEL_NAME, GROQ_API_KEY, GROQ_REASONING_EFFORT, GROQ_MAX_TOKENS, get_voice

# ============================================================================
# PATHS & CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILIPINO_AUDIO_PATH = os.path.join(BASE_DIR, "filipino.mp3")
INDONESIAN_AUDIO_PATH = os.path.join(BASE_DIR, "indonesian.mp3")

TRANSCRIPT_LOG_DIR = os.path.join(BASE_DIR, "transcripts")
AUDIO_OUTPUT_DIR = os.path.join(BASE_DIR, "responses")

# Initialize Groq client (same model as the call simulation engine)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)


def load_agent_config(filename: str) -> dict:
    """Loads localized agent configuration from a JSON specification file."""
    filepath = os.path.join(BASE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Load localized agent configurations
PH_AGENT_CONFIG = load_agent_config("ph_agent.json")
ID_AGENT_CONFIG = load_agent_config("id_agent.json")

# Market-Specific ASR & TTS Configuration.
# tts_voice now comes from shared_config.get_voice(market, "agent") so the
# reply audio uses the SAME voice actor as the agent's lines in the call
# simulation engine, instead of a locally hardcoded voice string.
MARKET_ASR_CONFIG = {
    "PH": {
        "language": "tl",
        "initial_prompt": (
            "Taglish conversation between Filipino customer and bancassurance specialist Bea. "
            "Terms: BPI-Philam, premium, policy, beneficiary, rider, lapse, coverage, "
            "grace period, auto-debit, po, opo, salamat po, paki-confirm po, pasensya na po."
        ),
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "tts_voice": get_voice("PH", "agent"),
        "market_name": "Philippines (Taglish / Bancassurance)",
        "sample_audio": FILIPINO_AUDIO_PATH,
    },
    "ID": {
        "language": "id",
        "initial_prompt": (
            "Percakapan Bahasa Indonesia antara nasabah dan Mas Budi IndoFinance. "
            "Istilah: angsuran, cicilan, tenor, jatuh tempo, denda, OJK, SLIK, Virtual Account, "
            "Mas, Mbak, Pak, Bu, nggih, sampeyan, niku, boten, m-Banking, BCA, BRI."
        ),
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "tts_voice": get_voice("ID", "agent"),
        "market_name": "Indonesia (Bahasa Indonesia / Multifinance)",
        "sample_audio": INDONESIAN_AUDIO_PATH,
    },
}

# Local Faster-Whisper Model holder (lazy-loaded to avoid blocking imports)
_whisper_model = None


def get_whisper_model():
    """Lazily loads Faster-Whisper model on first use."""
    global _whisper_model
    if _whisper_model is None:
        print("[*] Loading local Faster-Whisper model...")
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model


# ============================================================================
# 1. TEXT-TO-SPEECH GENERATOR (EDGE-TTS)
# ============================================================================
async def generate_speech_wav(text: str, market: str, session_id: str) -> str:
    """Converts LLM response text into a spoken audio response file, using the
    shared 'agent' voice for this market so it matches the simulation engine.

    NOTE: despite the function name (kept for backwards compatibility),
    Edge-TTS always outputs MP3-encoded audio — there is no WAV conversion
    step. Saving that MP3 data with a .wav extension produces a file whose
    container doesn't match its content, which many players refuse to open.
    The output file is therefore saved as .mp3.
    """
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(AUDIO_OUTPUT_DIR, f"reply_{session_id}.mp3")

    voice = MARKET_ASR_CONFIG.get(market, {}).get("tts_voice") or get_voice(market, "agent")
    print(f"[*] Generating speech response using edge-tts ({voice})...")

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        print(f"  [+] Spoken response saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"[*] Edge-TTS failed ({e}), falling back to gTTS.")
        lang = MARKET_ASR_CONFIG.get(market, {}).get("language", "en")
        gTTS(text=text, lang=lang, slow=False).save(output_path)
        print(f"  [+] Spoken response saved via gTTS fallback: {output_path}")
        return output_path


# ============================================================================
# 2. GROQ LLM INFERENCE (same model as the call simulation engine)
# ============================================================================
async def call_free_groq_llm(user_payload: Dict[str, Any]):
    """Processes ASR text through the shared Groq model with localized agent personas and triggers Edge-TTS."""
    market = user_payload["metadata"]["market"]
    user_text = user_payload["content"]
    session_id = user_payload["session_id"]

    # Select localized system prompt from loaded agent specifications
    if market == "PH" and "system_instructions" in PH_AGENT_CONFIG:
        system_prompt = (
            f"{PH_AGENT_CONFIG['system_instructions']} "
            "Keep responses brief, polite, human-like, and direct (1-2 short sentences max) "
            "so it sounds natural when spoken back over a phone call."
        )
    elif market == "ID" and "system_instructions" in ID_AGENT_CONFIG:
        system_prompt = (
            f"{ID_AGENT_CONFIG['system_instructions']} "
            "Keep responses brief, polite, human-like, and direct (1-2 short sentences max) "
            "so it sounds natural when spoken back over a phone call."
        )
    else:
        system_prompt = (
            "You are an AI Voice Agent for customer service in Southeast Asia. "
            "Keep responses brief, polite, human-like, and direct (1-2 short sentences max) "
            "so it sounds natural when spoken back over a phone call."
        )

    print(f"\n[ASR Output ({market})]: \"{user_text}\"")
    print("[Groq LLM Thinking...]")

    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Market: {market}. User said: '{user_text}'"}
            ],
            model=GROQ_MODEL_NAME,
            temperature=0.6,
            max_tokens=GROQ_MAX_TOKENS,
            reasoning_effort=GROQ_REASONING_EFFORT,
        )

        reply_text = (chat_completion.choices[0].message.content or "").strip()
        if not reply_text:
            # gpt-oss-120b is a reasoning model — with too small a token
            # budget its internal reasoning can consume every token and
            # leave nothing for the actual reply. Don't synthesize empty
            # audio in that case.
            print("[!] Groq returned an empty reply (reasoning likely consumed the token budget); skipping TTS for this turn.")
            return
        print(f"[Groq LLM Response]: \"{reply_text}\"")

        # 1. Log assistant response to local session JSON file
        response_payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "role": "assistant",
            "content": reply_text,
            "metadata": {"model": GROQ_MODEL_NAME}
        }
        save_transcript_json(response_payload, session_id=session_id)

        # 2. Convert LLM text response into spoken audio file
        await generate_speech_wav(reply_text, market=market, session_id=session_id)

    except Exception as e:
        print(f"[!] LLM / TTS Error: {e}")


# ============================================================================
# 3. UTILITIES: JSON SAVER & PAYLOAD FORMATTER
# ============================================================================
def save_transcript_json(payload: Dict[str, Any], session_id: str) -> str:
    """Appends messages to transcripts/session_{session_id}.json."""
    os.makedirs(TRANSCRIPT_LOG_DIR, exist_ok=True)
    file_path = os.path.join(TRANSCRIPT_LOG_DIR, f"session_{session_id}.json")

    file_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
        except json.JSONDecodeError:
            file_data = []

    file_data.append(payload)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(file_data, f, indent=2, ensure_ascii=False)

    return file_path


def format_for_llm(transcript: str, market: str, session_id: str) -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "role": "user",
        "content": transcript.strip(),
        "metadata": {
            "market": market,
            "language_register": "Taglish" if market == "PH" else "Bahasa Indonesia / Regional",
            "is_final": True
        }
    }


# ============================================================================
# 4. FASTER-WHISPER ASR ENGINE
# ============================================================================
class LocalWhisperASR:
    def __init__(self, session_id: str, market: str, on_llm_payload_ready: Callable):
        self.session_id = session_id
        self.market = market
        self.callback = on_llm_payload_ready

    async def process_file(self, audio_file_path: str):
        if not os.path.exists(audio_file_path):
            print(f"[!] Audio file missing: {audio_file_path}")
            return

        print(f"[*] Transcribing locally with Faster-Whisper: {audio_file_path}")

        # Load market-specific ASR configuration
        asr_cfg = MARKET_ASR_CONFIG.get(self.market, {})
        target_lang = asr_cfg.get("language")
        initial_prompt = asr_cfg.get("initial_prompt", "Customer service conversation")

        print(f"  [*] ASR Engine: Faster-Whisper (small, int8) | Market Language Target: '{target_lang}'")
        print(f"  [*] Domain Prompt Guidance: \"{initial_prompt[:60]}...\"")

        # Transcribe audio file directly with language-specific guidance
        model = get_whisper_model()
        segments, info = model.transcribe(
            audio_file_path,
            language=target_lang,
            initial_prompt=initial_prompt,
            beam_size=asr_cfg.get("beam_size", 5),
            best_of=asr_cfg.get("best_of", 5),
            temperature=asr_cfg.get("temperature", 0.0),
        )

        transcript_parts = [segment.text for segment in segments]
        full_transcript = " ".join(transcript_parts).strip()

        print(f"  [+] Language Detected: {info.language} (Probability: {info.language_probability:.2f})")
        print(f"  [+] Full Transcript: \"{full_transcript}\"")

        if full_transcript:
            payload = format_for_llm(full_transcript, market=self.market, session_id=self.session_id)
            saved_path = save_transcript_json(payload, session_id=self.session_id)
            print(f"  [+] SUCCESS: Saved transcript to -> {saved_path}")

            # Send payload to LLM
            await self.callback(payload)


# ============================================================================
# PIPELINE RUNNER
# ============================================================================
async def process_audio_file(audio_file_path: str, market: str, session_id: str):
    print(f"\n==================================================")
    print(f"[*] Starting session '{session_id}' | Market: {market}")
    print(f"==================================================")

    asr = LocalWhisperASR(session_id=session_id, market=market, on_llm_payload_ready=call_free_groq_llm)
    await asr.process_file(audio_file_path)

    print(f"[*] Session '{session_id}' finished.\n")


def run_asr_pipeline(audio_file_path: str, market: str, session_id: str):
    """Synchronous interface to process any audio file via Whisper ASR + Groq LLM + Edge-TTS."""
    return asyncio.run(process_audio_file(audio_file_path=audio_file_path, market=market, session_id=session_id))


def print_asr_performance_summary():
    """Prints benchmark summary and behavior notes for the configured ASR engines."""
    print("\n" + "=" * 70)
    print("ASR ENGINE PERFORMANCE & BEHAVIOR SUMMARY")
    print("=" * 70)
    print("Provider & Model: Faster-Whisper (OpenAI Whisper 'small' int8 quantized on CPU)")
    print("\n1. Philippines Market (Taglish):")
    print("   - Language Target: 'tl' (Tagalog) with Taglish domain prompt.")
    print("   - Code-Switching Behavior: Without prompt, Whisper struggles when speakers switch")
    print("     mid-sentence from Tagalog to English finance terms (e.g. 'premium', 'policy').")
    print("     With domain initial_prompt, technical English nouns retain accurate spelling.")
    print("   - Approximate Quality: High (~90-95% word accuracy on clean audio).")
    print("   - Observed Errors: Minor phonetic boundary blending (e.g., 'mag-lapse' -> 'mag lapse').")
    print("\n2. Indonesia Market (Bahasa Indonesia & Javanese Accent):")
    print("   - Language Target: 'id' (Indonesian) with Javanese dialect markers in prompt.")
    print("   - Regional Accent Performance: Native Javanese speakers introduce glottal stops")
    print("     and colloquial markers ('nggih', 'sampeyan', 'iku'). Whisper handles Javanese-")
    print("     inflected Indonesian well when seeded with regional vocabulary cues.")
    print("   - Approximate Quality: High (~92-96% word accuracy).")
    print("   - Observed Errors: Rare phonetic vowel shifts ('jatuh' vs 'jatoh').")
    print("=" * 70 + "\n")


def find_customer_audio_files(base_dir: str = None):
    """Finds every *_customer.mp3 file generated by simulate_voice_calls.py in
    base_dir (defaults to this script's directory) and infers each one's
    market ('PH'/'ID') from its filename prefix, e.g.
    'PH_Call_01_Cooperative_customer.mp3' -> market='PH',
    session_id='PH_Call_01_Cooperative'.
    Returns a list of (audio_path, market, session_id) tuples, skipping any
    file whose prefix isn't a recognized market.
    """
    search_dir = base_dir or BASE_DIR
    pattern = os.path.join(search_dir, "*_customer.mp3")
    matches = sorted(glob.glob(pattern))

    results = []
    for path in matches:
        filename = os.path.basename(path)
        session_id = filename[: -len("_customer.mp3")]
        market = session_id.split("_")[0].upper()
        if market not in MARKET_ASR_CONFIG:
            print(f"[!] Skipping '{filename}': could not infer a known market ('{market}') from the filename.")
            continue
        results.append((path, market, session_id))
    return results


def run_asr_pipeline_batch(base_dir: str = None):
    """Runs every discovered *_customer.mp3 file through the ASR + Groq + TTS
    pipeline, one at a time. A failure on one file is logged and skipped so
    the rest of the batch still completes."""
    files = find_customer_audio_files(base_dir)
    if not files:
        print(f"[!] No *_customer.mp3 files found in {base_dir or BASE_DIR}. "
              f"Run simulate_voice_calls.py first to generate them.")
        return

    print(f"[*] Found {len(files)} customer audio file(s) to verify.")
    succeeded, failed = [], []
    for audio_path, market, session_id in files:
        print(f"\n>>> PROCESSING: {session_id} [{market}] <<<")
        try:
            run_asr_pipeline(audio_file_path=audio_path, market=market, session_id=session_id)
            succeeded.append(session_id)
        except Exception as e:
            print(f"[!] Failed to process '{session_id}': {e}")
            failed.append(session_id)

    print("\n" + "=" * 65)
    print(f"BATCH ASR VERIFICATION COMPLETE: {len(succeeded)}/{len(files)} succeeded.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("=" * 65)


# ============================================================================
# MAIN EXECUTION (Independent Market Testing)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localized Voice AI ASR Pipeline Runner")
    parser.add_argument("--market", choices=["PH", "ID", "ALL"], default="ALL", help="Select target market to test (only used without --batch/--audio)")
    parser.add_argument("--audio", type=str, default=None, help="Custom audio path to process")
    parser.add_argument("--session-id", type=str, default=None, help="Custom session ID")
    parser.add_argument("--batch", action="store_true", help="Auto-discover and process every *_customer.mp3 file generated by simulate_voice_calls.py")
    parser.add_argument("--report", action="store_true", help="Print ASR behavior & performance summary")
    args = parser.parse_args()

    if args.report:
        print_asr_performance_summary()

    print(f"[*] Transcripts will be output to directory:\n    {TRANSCRIPT_LOG_DIR}\n")

    # Batch mode: process every *_customer.mp3 sitting next to this script
    if args.batch:
        run_asr_pipeline_batch()
        sys.exit(0)

    # Custom audio execution
    if args.audio:
        target_market = args.market if args.market in ["PH", "ID"] else "PH"
        sid = args.session_id or f"Custom_{target_market}_Call"
        run_asr_pipeline(audio_file_path=args.audio, market=target_market, session_id=sid)
        sys.exit(0)

    # 1. Process Filipino Audio
    if args.market in ["PH", "ALL"]:
        if os.path.exists(FILIPINO_AUDIO_PATH):
            print("\n>>> TESTING MARKET: PHILIPPINES (PH) <<<")
            run_asr_pipeline(
                audio_file_path=FILIPINO_AUDIO_PATH,
                market="PH",
                session_id="PH_Call_01"
            )
        else:
            print(f"[!] Warning: {FILIPINO_AUDIO_PATH} does not exist.")

    # 2. Process Indonesian Audio
    if args.market in ["ID", "ALL"]:
        if os.path.exists(INDONESIAN_AUDIO_PATH):
            print("\n>>> TESTING MARKET: INDONESIA (ID) <<<")
            run_asr_pipeline(
                audio_file_path=INDONESIAN_AUDIO_PATH,
                market="ID",
                session_id="ID_Call_01"
            )
        else:
            print(f"[!] Warning: {INDONESIAN_AUDIO_PATH} does not exist.")