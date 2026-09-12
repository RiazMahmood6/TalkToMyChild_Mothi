"""All external services in one file.

  STT  -> Groq Whisper               transcribe()
  LLM  -> Groq Llama (chat + stream) chat()  /  chat_stream()
  TTS  -> Cartesia Sonic             synthesize()  /  synthesize_stream()
  CLONE-> Cartesia voice clone       clone_voice()
  MEM  -> Groq summariser            summarize_conversation()

Why Cartesia replaces RunPod:
  * ~40ms time-to-first-audio (vs RunPod's batch job + download round-trip)
  * Clones a voice ONCE -> returns a voice_id you store on the Character.
    Every later call just references that id. No ngrok / PUBLIC_BASE_URL,
    because Cartesia never has to reach back into your server for the
    reference clip.
  * True streaming (SSE) so the kid hears the reply start almost instantly.

LEGAL NOTE: voice cloning here is for CONSENTING people (a parent/relative
who recorded their own sample). Cloning a non-consenting person — celebrity
or otherwise — is a legal problem regardless of provider. Keep a consent
record alongside every cloned voice_id.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Iterator, Optional

import requests
from groq import Groq

from config import Config

log = logging.getLogger(__name__)

CARTESIA_BASE = "https://api.cartesia.ai"
_groq_client: Optional[Groq] = None


# Shared persona for "quick chat" mode (no character selected) — used by
# both the live voice agent (voice_agent.py --generic) and the text-only
# chat endpoint (/api/chat/generic), so the assistant feels the same
# whether the child is talking or typing.
GENERIC_SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant talking with a child. "
    "Speak warmly, in first person. Keep replies SHORT — 1 to 3 sentences. "
    "Only use facts the child actually tells you — do not invent details "
    "about yourself or them. Avoid scary, violent, sexual, or "
    "age-inappropriate content. If the child seems upset or in danger, "
    "gently suggest a trusted grown-up. No emojis, no markdown, no stage "
    "directions — only spoken words."
)


def _groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not Config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=Config.GROQ_API_KEY)
    return _groq_client


def _cartesia_headers(json_body: bool = True) -> dict:
    if not Config.CARTESIA_API_KEY:
        raise RuntimeError("CARTESIA_API_KEY not set in .env")
    h = {
        "Authorization": f"Bearer {Config.CARTESIA_API_KEY}",
        "Cartesia-Version": Config.CARTESIA_VERSION,
    }
    if json_body:
        h["Content-Type"] = "application/json"
    return h


# ============================================================
# STT  — Groq Whisper  (unchanged behaviour)
# ============================================================
def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    if not audio_bytes:
        return ""
    result = _groq().audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=Config.GROQ_STT_MODEL,
        response_format="text",
        temperature=0.0,
    )
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return (text or "").strip()


# ============================================================
# LLM  — Groq Llama
# ============================================================
def _build_messages(system_prompt: str, history: List[Dict[str, str]],
                    user_msg: str) -> list:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    return messages


def chat(system_prompt: str, history: List[Dict[str, str]], user_msg: str) -> str:
    """Non-streaming reply (kept for the simple turn-based endpoints)."""
    resp = _groq().chat.completions.create(
        model=Config.GROQ_LLM_MODEL,
        messages=_build_messages(system_prompt, history, user_msg),
        temperature=0.3,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def chat_stream(system_prompt: str, history: List[Dict[str, str]],
                user_msg: str) -> Iterator[str]:
    """Yields reply tokens as they arrive. Feed these into a sentence
    aggregator so TTS can start on the first finished sentence — this is
    what makes the call feel live rather than turn-based.
    """
    stream = _groq().chat.completions.create(
        model=Config.GROQ_LLM_MODEL,
        messages=_build_messages(system_prompt, history, user_msg),
        temperature=0.3,
        max_tokens=200,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


# ============================================================
# TTS  — Cartesia Sonic
# ============================================================
def _voice_block(voice_id: Optional[str]) -> dict:
    vid = voice_id or Config.CARTESIA_FALLBACK_VOICE_ID
    return {"mode": "id", "id": vid}


def synthesize(text: str, voice_id: Optional[str]) -> bytes:
    """One-shot: returns a full WAV. Used by preview / simple endpoints.

    `voice_id` is the Cartesia voice id stored on the character
    (from clone_voice()). If None, falls back to the preset voice.
    """
    payload = {
        "model_id": Config.CARTESIA_TTS_MODEL,
        "transcript": text,
        "voice": _voice_block(voice_id),
        "language": Config.CARTESIA_LANGUAGE,
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": Config.CARTESIA_SAMPLE_RATE,
        },
    }
    log.info("Cartesia TTS bytes: %r (voice=%s)", text[:60],
             voice_id or "PRESET")
    r = requests.post(f"{CARTESIA_BASE}/tts/bytes",
                      json=payload, headers=_cartesia_headers(), timeout=60)
    if r.status_code == 401:
        raise RuntimeError("Cartesia API key invalid")
    if r.status_code != 200:
        raise RuntimeError(f"Cartesia HTTP {r.status_code}: {r.text[:300]}")
    return r.content


def synthesize_stream(text: str, voice_id: Optional[str]) -> Iterator[bytes]:
    """Streams raw PCM audio chunks via SSE as they're generated.

    Use this in the live pipeline: pass one sentence at a time and pipe
    the yielded chunks straight to the client for immediate playback.
    """
    payload = {
        "model_id": Config.CARTESIA_TTS_MODEL,
        "transcript": text,
        "voice": _voice_block(voice_id),
        "language": Config.CARTESIA_LANGUAGE,
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": Config.CARTESIA_SAMPLE_RATE,
        },
    }
    with requests.post(f"{CARTESIA_BASE}/tts/sse", json=payload,
                       headers=_cartesia_headers(), stream=True,
                       timeout=60) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Cartesia HTTP {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8", "ignore")
            if decoded.startswith("data:"):
                import base64
                import json as _json
                try:
                    evt = _json.loads(decoded[5:].strip())
                except Exception:
                    continue
                b64 = evt.get("data")
                if b64:
                    yield base64.b64decode(b64)


# ============================================================
# CLONE  — Cartesia instant voice clone
# ============================================================
def clone_voice(audio_path: str, name: str,
                description: str = "") -> str:
    """Clone a voice from a short sample (~5-20s). Returns a voice_id
    to store on the Character. Call this ONCE when a character is created.
    """
    with open(audio_path, "rb") as fh:
        files = {"clip": (name, fh, "application/octet-stream")}
        data = {
            "name": name[:60] or "voice",
            "description": (description or f"Cloned voice for {name}")[:200],
            "language": Config.CARTESIA_LANGUAGE,
            "mode": "clip",
        }
        r = requests.post(f"{CARTESIA_BASE}/voices/clone",
                          headers=_cartesia_headers(json_body=False),
                          files=files, data=data, timeout=120)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Cartesia clone HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    voice_id = body.get("id") or (body.get("voice") or {}).get("id")
    if not voice_id:
        raise RuntimeError(f"Clone response missing voice id: {body}")
    log.info("Cloned voice %r -> %s", name, voice_id)
    return voice_id


# ============================================================
# MEMORY  — summarise a finished conversation for long-term recall
# ============================================================
_SUMMARY_SYSTEM = (
    "You maintain a running memory for a character who talks with a child "
    "on phone calls. Merge the EXISTING MEMORY with the NEW CONVERSATION "
    "into one updated memory. Keep it short (under 150 words). Preserve "
    "stable facts the character should always remember: the child's name, "
    "family, pets, school, friends, favourite things, and anything the "
    "child cared about. Drop small talk. Write it as plain notes addressed "
    "to the character, e.g. 'The child's name is Mia. She has a dog named "
    "Rex.' No preamble, just the notes."
)


def summarize_conversation(existing_memory: str,
                           history: List[Dict[str, str]]) -> str:
    """Roll the existing memory + this call's transcript into one updated
    memory string. Run this when a call ends; store the result on the
    character; inject it into the system prompt next call.
    """
    if not history:
        return existing_memory or ""
    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in history
        if m.get("role") in ("user", "assistant")
    )
    user_block = (
        f"EXISTING MEMORY:\n{existing_memory or '(none yet)'}\n\n"
        f"NEW CONVERSATION:\n{transcript}\n\n"
        f"Updated memory:"
    )
    resp = _groq().chat.completions.create(
        model=Config.GROQ_SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": user_block},
        ],
        temperature=0.3,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()
