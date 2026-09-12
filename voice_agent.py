"""Real-time voice-call agent (Pipecat 1.3.0).

  Browser mic  ──ws──>  [VAD] ─> Groq Whisper STT ─> Groq LLM ─> Cartesia TTS ──ws──> Browser speaker

FIXES (verified against pipecat 1.3.0 source):
  * RawPCMSerializer — call.html speaks raw s16le PCM, not protobuf.
  * VADProcessor in the pipeline — in 1.3.0 vad_analyzer is NOT a
    transport param anymore (it was silently ignored, so VAD never ran,
    so the segmented Whisper STT never transcribed → bot greeted but
    couldn't hear you). VAD now runs as its own pipeline stage.
  * allow_interruptions removed from PipelineParams (gone in 1.3.0);
    barge-in is automatic via the turn system.
  * 16 kHz in/out everywhere to match the browser AudioContext.

Run one agent bound to a character:
    python voice_agent.py --char 1 --port 8765
"""
from __future__ import annotations
import argparse
import asyncio
import logging
from typing import Optional

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from pipecat.transcriptions.language import Language

from config import Config
import tts_stt_llm as svc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("voice_agent")

# Must match call.html  (CONFIG.sampleRate = 16000)
BROWSER_SAMPLE_RATE = 16000


# ----------------------------------------------------------------
# Raw PCM serializer — browser <-> pipecat without protobuf.
#   incoming ws binary  -> InputAudioRawFrame (mic, s16le mono 16k)
#   OutputAudioRawFrame -> raw bytes (speaker)
# Everything else (text/control frames) is dropped: the browser
# only understands PCM.
# ----------------------------------------------------------------
class RawPCMSerializer(FrameSerializer):
    def __init__(self, sample_rate: int = BROWSER_SAMPLE_RATE, **kwargs):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate

    async def serialize(self, frame: Frame) -> Optional[bytes]:
        if isinstance(frame, OutputAudioRawFrame):
            return bytes(frame.audio)
        return None

    async def deserialize(self, data) -> Optional[Frame]:
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(
                audio=bytes(data),
                sample_rate=self._sample_rate,
                num_channels=1,
            )
        return None


# ----------------------------------------------------------------
# Drops empty / too-short / hallucinated transcripts before they
# reach the LLM. Whisper invents phrases like "thank you" or "bye"
# when fed silence or background noise — this filters those out.
# ----------------------------------------------------------------
_HALLUCINATION_BLOCKLIST = {
    "thank you", "thank you.", "thanks for watching", "bye", "bye.",
    "you", ".", "..", "...", "hmm", "hmm.", "okay", "ok",
}


class TranscriptGuard(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            normalized = text.lower()

            if not text or len(text) < 3 or normalized in _HALLUCINATION_BLOCKLIST:
                log.info("Dropped suspected hallucinated transcript: %r", text)
                return  # swallow the frame — don't forward it downstream

        await self.push_frame(frame, direction)


# ----------------------------------------------------------------
# Load character data from the SAME db the Flask app uses.
# ----------------------------------------------------------------
def load_character(char_id: int) -> dict:
    from app import app  # reuse the configured Flask app + db
    from models import Character
    with app.app_context():
        char = Character.query.get(char_id)
        if not char:
            raise SystemExit(f"No character with id {char_id}")
        return {
            "id": char.id,
            "name": char.name,
            "voice_id": char.voice_id or Config.CARTESIA_FALLBACK_VOICE_ID,
            "system_prompt": char.system_prompt(),  # already includes memory
            "memory": char.memory or "",
        }


def load_generic() -> dict:
    """Plain assistant persona — no character, no memory, preset voice.
    Used for --generic mode: a bare Groq chat + generic Cartesia voice,
    with no DB lookup and nothing persisted afterward. Shares its system
    prompt with the text-only /api/chat/generic endpoint in app.py so the
    assistant is consistent across voice and text.
    """
    return {
        "id": None,
        "name": "Assistant",
        "voice_id": Config.CARTESIA_FALLBACK_VOICE_ID,
        "system_prompt": svc.GENERIC_SYSTEM_PROMPT,
        "memory": "",
    }


def save_transcript(char_id: int, history: list[dict]) -> None:
    """On hangup: store the raw turn-by-turn transcript of this call so
    it can be retrieved later via GET /api/characters/<id>/transcript.
    No-ops for generic-mode calls (char_id is None there).
    """
    if not history:
        return
    import json
    from app import app
    from models import Character, db
    try:
        with app.app_context():
            char = Character.query.get(char_id)
            if char:
                char.last_transcript = json.dumps(history)
                db.session.commit()
                log.info("Transcript saved for character %s", char_id)
    except Exception:
        log.exception("Failed to save transcript")


def persist_memory(char_id: int, existing_memory: str,
                   history: list[dict]) -> None:
    """On hangup: roll this call into the character's long-term memory."""
    if not history:
        return
    from app import app
    from models import Character, db
    try:
        new_memory = svc.summarize_conversation(existing_memory, history)
        with app.app_context():
            char = Character.query.get(char_id)
            if char:
                char.memory = new_memory
                db.session.commit()
                log.info("Memory updated for character %s", char_id)
    except Exception:
        log.exception("Failed to persist memory")


async def run_bot(char: dict, host: str, port: int) -> None:
    transport = WebsocketServerTransport(
        host=host,
        port=port,
        params=WebsocketServerParams(
            serializer=RawPCMSerializer(BROWSER_SAMPLE_RATE),   # <-- FIX 1
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=BROWSER_SAMPLE_RATE,           # <-- FIX 2
            audio_out_sample_rate=BROWSER_SAMPLE_RATE,          # <-- FIX 2
            add_wav_header=False,
        ),
    )

    stt = GroqSTTService(
        api_key=Config.GROQ_API_KEY,
        settings=GroqSTTService.Settings(
            model=Config.GROQ_STT_MODEL,
            language=Language.EN,
        ),
    )

    llm = GroqLLMService(
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_LLM_MODEL,
    )

    tts = CartesiaTTSService(
        api_key=Config.CARTESIA_API_KEY,
        voice_id=char["voice_id"],
        model=Config.CARTESIA_TTS_MODEL,
        params=CartesiaTTSService.InputParams(language=Language.EN),
    )

    context = LLMContext(
        messages=[{"role": "system", "content": char["system_prompt"]}],
    )
    context_aggregator = LLMContextAggregatorPair(context)

    # In pipecat 1.3.0, VAD is a pipeline processor (no longer a transport
    # param — passing vad_analyzer to the transport is silently ignored,
    # which is why STT never fired). VADProcessor emits the speech
    # start/stop frames that drive Whisper STT, turn-taking and barge-in.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.75,   # default is lower; raise to ignore soft noise
                min_volume=0.6,
                start_secs=0.3,    # require sustained speech before triggering
                stop_secs=0.8,     # require sustained silence before ending turn
            )
        )
    )

    # Filters out empty/too-short/hallucinated Whisper transcripts before
    # they're added to the LLM context as if the child actually said them.
    transcript_guard = TranscriptGuard()

    pipeline = Pipeline([
        transport.input(),
        vad,                        # speech detection (drives STT + barge-in)
        stt,
        transcript_guard,           # <-- drops junk transcripts here
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            # NOTE: allow_interruptions no longer exists in pipecat 1.3.0 —
            # barge-in now happens automatically via the turn system.
            audio_in_sample_rate=BROWSER_SAMPLE_RATE,
            audio_out_sample_rate=BROWSER_SAMPLE_RATE,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        log.info("Child connected — %s greeting", char["name"])
        if char["id"] is not None:
            greeting = (f"Hi sweetheart, it's {char['name']}. "
                        f"How are you doing today?")
        else:
            greeting = "Hi there! How can I help you today?"
        await task.queue_frame(TTSSpeakFrame(greeting))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        messages = context.get_messages()
        history = [m for m in messages if m.get("role") in ("user", "assistant")]
        if char["id"] is not None:
            log.info("Child disconnected — saving transcript + memory")
            save_transcript(char["id"], history)
            persist_memory(char["id"], char["memory"], history)
        else:
            log.info("Child disconnected — generic mode, nothing to save")
        await task.cancel()

    runner = PipelineRunner()
    log.info("Voice agent for %r live on ws://%s:%d", char["name"], host, port)
    await runner.run(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", type=int, required=False, help="Character id")
    ap.add_argument("--generic", action="store_true",
                    help="Run a plain Groq chat + generic voice, no "
                         "character/memory (ignores --char if both given)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not Config.GROQ_API_KEY or not Config.CARTESIA_API_KEY:
        raise SystemExit("Set GROQ_API_KEY and CARTESIA_API_KEY in .env first.")

    if args.generic:
        char = load_generic()
    else:
        if args.char is None:
            raise SystemExit("Pass --char <id>, or use --generic for no character.")
        char = load_character(args.char)

    asyncio.run(run_bot(char, args.host, args.port))


if __name__ == "__main__":
    main()