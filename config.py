"""Centralised config — read from .env once."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-only")
    DEBUG = os.getenv("FLASK_ENV", "production") == "development"

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / 'instance' / 'dadvoice.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = str(BASE_DIR / "static" / "uploads")
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB

    # ---- Groq: STT + LLM (unchanged) ----
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    # llama-3.1-8b-instant was decommissioned by Groq on 2026-08-16.
    # openai/gpt-oss-20b is Groq's own recommended replacement: similar
    # size/speed class, generally stronger quality.
    GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")
    GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    # Used only for the offline memory-summarisation step. Latency doesn't
    # matter there, so you *may* point this at a bigger model later.
    GROQ_SUMMARY_MODEL = os.getenv("GROQ_SUMMARY_MODEL", "openai/gpt-oss-20b")

    # ---- Cartesia: TTS + voice cloning (replaces RunPod) ----
    # No PUBLIC_BASE_URL / ngrok needed anymore: Cartesia clones once and
    # gives back a voice_id we store, instead of fetching reference audio
    # from a public URL on every call.
    CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
    CARTESIA_VERSION = os.getenv("CARTESIA_VERSION", "2024-11-13")
    CARTESIA_TTS_MODEL = os.getenv("CARTESIA_TTS_MODEL", "sonic-2")
    # Fallback preset voice id used when a character has no clone yet.
    CARTESIA_FALLBACK_VOICE_ID = os.getenv(
        "CARTESIA_FALLBACK_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"
    )
    CARTESIA_LANGUAGE = os.getenv("CARTESIA_LANGUAGE", "en")
    CARTESIA_SAMPLE_RATE = int(os.getenv("CARTESIA_SAMPLE_RATE", "44100"))

    # Where the Pipecat live-call agent (voice_agent.py) listens. The frontend
    # call screen connects here. For local dev this is the default; in
    # production the backend would spawn an agent and hand back its URL.
    AGENT_WS_URL = os.getenv("AGENT_WS_URL", "ws://localhost:8765")
