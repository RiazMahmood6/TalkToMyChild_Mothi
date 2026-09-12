"""TalkToMyChild — API. Cartesia TTS + cloning, Groq STT/LLM, rolling memory.

Run:  python app.py
Health: http://localhost:5000/api/health

What changed from the RunPod version:
  * TTS + cloning now use Cartesia (see tts_stt_llm.py). No ngrok needed.
  * Characters are cloned ONCE at creation; we store voice_id on them.
  * Conversations are summarised into Character.memory when a call ends,
    and that memory is injected into the system prompt on the next call.
  * New /turn_stream endpoint pipelines STT -> streaming LLM -> streaming
    TTS so the reply starts playing almost immediately (the "live" feel).
"""
from __future__ import annotations
import logging
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from flask import (Flask, request, jsonify, send_file, send_from_directory,
                   Response, stream_with_context, render_template)

from config import Config
from models import db, User, Character
import tts_stt_llm as svc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dadvoice")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)

os.makedirs(app.instance_path, exist_ok=True)
os.makedirs(Path(Config.UPLOAD_FOLDER) / "voices", exist_ok=True)
os.makedirs(Path(Config.UPLOAD_FOLDER) / "images", exist_ok=True)

db.init_app(app)
with app.app_context():
    db.create_all()


ALLOWED_AUDIO = {"mp3", "wav", "ogg", "webm", "m4a", "flac"}
ALLOWED_IMAGE = {"png", "jpg", "jpeg", "webp", "gif"}


def _current_user() -> User:
    user = User.query.first()
    if not user:
        user = User(name="Demo")
        db.session.add(user)
        db.session.commit()
    return user


def _save_upload(file_storage, allowed: set, subdir: str):
    if not file_storage or not file_storage.filename:
        return None
    ext = file_storage.filename.rsplit(".", 1)[-1].lower() if "." in file_storage.filename else ""
    if ext not in allowed:
        raise ValueError(f"Unsupported file type: .{ext} (allowed: {sorted(allowed)})")
    out_dir = Path(Config.UPLOAD_FOLDER) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.{ext}"
    file_storage.save(out_dir / fname)
    return f"uploads/{subdir}/{fname}"


# ============================================================
# Health
# ============================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "groq_key_set": bool(Config.GROQ_API_KEY),
        "cartesia_key_set": bool(Config.CARTESIA_API_KEY),
        "cloning_enabled": bool(Config.CARTESIA_API_KEY),
        "groq_llm_model": Config.GROQ_LLM_MODEL,
        "groq_stt_model": Config.GROQ_STT_MODEL,
        "cartesia_tts_model": Config.CARTESIA_TTS_MODEL,
        "agent_ws_url": Config.AGENT_WS_URL,
    })


# ============================================================
# Frontend pages
# ============================================================
@app.route("/", methods=["GET"])
def home():
    """Character manager: list, create, preview, and launch a call."""
    return render_template("index.html")


@app.route("/call", methods=["GET"])
def call_page():
    """Live call screen. Open with ?char=<id> (and optionally &ws=...)."""
    return render_template("call.html")


@app.route("/quickchat", methods=["GET"])
def quickchat_page():
    """Quick chat: talk to a plain assistant with no character selected.
    Offers both a text chat box and a button to switch to a live voice
    call (?generic=1 on the /call screen)."""
    return render_template("quickchat.html")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# ============================================================
# Isolated test endpoints
# ============================================================
@app.route("/api/test/stt", methods=["POST"])
def test_stt():
    if "audio" not in request.files:
        return jsonify({"error": "Upload a file as multipart field 'audio'"}), 400
    f = request.files["audio"]
    audio_bytes = f.read()
    try:
        text = svc.transcribe(audio_bytes, filename=f.filename or "audio.webm")
        return jsonify({"text": text, "bytes": len(audio_bytes)})
    except Exception as e:
        log.exception("test/stt failed")
        return jsonify({"error": str(e)}), 502


@app.route("/api/test/llm", methods=["POST"])
def test_llm():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Send JSON {\"prompt\": \"...\"}"}), 400
    system = data.get("system") or "You are a friendly assistant. Keep replies short."
    try:
        return jsonify({"reply": svc.chat(system, [], prompt)})
    except Exception as e:
        log.exception("test/llm failed")
        return jsonify({"error": str(e)}), 502


@app.route("/api/test/tts", methods=["POST"])
def test_tts():
    """JSON: {"text": "...", "voice_id": "(optional Cartesia id)"}"""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Send JSON {\"text\": \"...\"}"}), 400
    try:
        audio = svc.synthesize(text, data.get("voice_id"))
    except Exception as e:
        log.exception("test/tts failed")
        return jsonify({"error": str(e)}), 502
    return send_file(BytesIO(audio), mimetype="audio/wav",
                     download_name="tts.wav", as_attachment=False)


# ============================================================
# Character CRUD
# ============================================================
@app.route("/api/characters", methods=["GET"])
def list_characters():
    user = _current_user()
    chars = Character.query.filter_by(user_id=user.id) \
        .order_by(Character.created_at.desc()).all()
    return jsonify([c.to_dict() for c in chars])


@app.route("/api/characters/<int:char_id>", methods=["GET"])
def get_character(char_id):
    return jsonify(Character.query.get_or_404(char_id).to_dict())


@app.route("/api/characters", methods=["POST"])
def create_character():
    """Multipart: name (req), age/gender/role/category/description (opt),
    voice_sample (req audio), image (opt). Clones the voice via Cartesia
    and stores the resulting voice_id.
    """
    user = _current_user()
    f = request.form
    name = (f.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Field 'name' is required"}), 400

    try:
        image_path = _save_upload(request.files.get("image"), ALLOWED_IMAGE, "images")
        voice_path = _save_upload(request.files.get("voice_sample"), ALLOWED_AUDIO, "voices")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not voice_path:
        return jsonify({"error": "Field 'voice_sample' (audio file) is required"}), 400

    # Clone once, now, while we have the file on disk.
    voice_id = None
    abs_path = Path(app.static_folder) / voice_path
    try:
        voice_id = svc.clone_voice(str(abs_path), name=name,
                                   description=f.get("description", "") or "")
    except Exception as e:
        log.warning("Clone failed, will use preset voice: %s", e)

    char = Character(
        user_id=user.id,
        name=name,
        age=f.get("age", "").strip() or None,
        gender=f.get("gender", "").strip() or None,
        role=f.get("role", "").strip() or None,
        category=f.get("category", "").strip() or None,
        description=f.get("description", "").strip() or None,
        image_path=image_path,
        voice_sample_path=voice_path,
        voice_id=voice_id,
        memory="",
    )
    db.session.add(char)
    db.session.commit()
    return jsonify(char.to_dict()), 201


@app.route("/api/characters/<int:char_id>", methods=["DELETE"])
def delete_character(char_id):
    char = Character.query.get_or_404(char_id)
    for rel in (char.image_path, char.voice_sample_path):
        if rel:
            try:
                (Path(app.static_folder) / rel).unlink(missing_ok=True)
            except Exception:
                pass
    db.session.delete(char)
    db.session.commit()
    return jsonify({"ok": True, "deleted": char_id})


@app.route("/api/characters/<int:char_id>/preview", methods=["POST"])
def preview_voice(char_id):
    char = Character.query.get_or_404(char_id)
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip() or \
           f"Hi! It's {char.name}. I'm so happy to talk with you today."
    try:
        audio = svc.synthesize(text, char.voice_id)
    except Exception as e:
        log.exception("preview failed")
        return jsonify({"error": str(e)}), 502
    return send_file(BytesIO(audio), mimetype="audio/wav", download_name="preview.wav")


# ============================================================
# Call API
# ============================================================
_call_history: dict[int, list[dict]] = {}  # MVP in-memory; move to Redis later


@app.route("/api/call/<int:char_id>/start", methods=["POST"])
def call_start(char_id):
    char = Character.query.get_or_404(char_id)
    _call_history[char_id] = []
    return jsonify({"ok": True, "character": char.to_dict()})


@app.route("/api/call/<int:char_id>/greeting", methods=["POST"])
def call_greeting(char_id):
    char = Character.query.get_or_404(char_id)
    _call_history.setdefault(char_id, [])
    text = f"Hi sweetheart, it's {char.name}. How are you doing today?"
    try:
        audio = svc.synthesize(text, char.voice_id)
    except Exception as e:
        log.exception("greeting tts failed")
        return jsonify({"error": str(e)}), 502
    _call_history[char_id].append({"role": "assistant", "content": text})
    return _audio_with_text(audio, reply=text, user_text="")


@app.route("/api/call/<int:char_id>/turn", methods=["POST"])
def call_turn(char_id):
    """Simple (non-streaming) full turn — kept for easy Postman testing."""
    char = Character.query.get_or_404(char_id)
    if "audio" not in request.files:
        return jsonify({"error": "Upload audio as multipart field 'audio'"}), 400
    audio_bytes = request.files["audio"].read()
    if len(audio_bytes) < 1000:
        return jsonify({"error": "Audio too short (< 1KB)"}), 400

    try:
        user_text = svc.transcribe(audio_bytes, filename="audio.webm")
    except Exception as e:
        return jsonify({"error": f"STT failed: {e}"}), 502
    if not user_text or len(user_text) < 2:
        return jsonify({"error": "Didn't catch any speech", "user_text": user_text}), 200

    history = _call_history.setdefault(char_id, [])
    try:
        reply = svc.chat(char.system_prompt(), history, user_text)
    except Exception as e:
        return jsonify({"error": f"LLM failed: {e}", "user_text": user_text}), 502

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 20:
        _call_history[char_id] = history[-20:]

    try:
        audio_out = svc.synthesize(reply, char.voice_id)
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}", "user_text": user_text,
                        "reply_text": reply}), 502
    return _audio_with_text(audio_out, reply=reply, user_text=user_text)


@app.route("/api/call/<int:char_id>/message", methods=["POST"])
def call_message(char_id):
    """Text-only turn — no audio in, no audio out. JSON {"text": "..."}
    -> {"reply": "..."}. Shares the same _call_history as the audio-based
    /turn endpoint, so a conversation can mix typed and spoken turns.
    This is what the call screen's 'Text chat' toggle uses.
    """
    char = Character.query.get_or_404(char_id)
    data = request.get_json(silent=True) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"error": "Send JSON {\"text\": \"...\"}"}), 400

    history = _call_history.setdefault(char_id, [])
    try:
        reply = svc.chat(char.system_prompt(), history, user_text)
    except Exception as e:
        return jsonify({"error": f"LLM failed: {e}", "user_text": user_text}), 502

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 20:
        _call_history[char_id] = history[-20:]

    return jsonify({"user_text": user_text, "reply": reply})


@app.route("/api/chat/generic", methods=["POST"])
def chat_generic():
    """Text-only chat with NO character — the 'Quick chat' page.

    Stateless on the server: the client keeps the running conversation
    and sends it back each turn. JSON in:
      {"text": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}
    JSON out: {"reply": "..."}

    Uses the same persona as `voice_agent.py --generic` (GENERIC_SYSTEM_PROMPT
    in tts_stt_llm.py), so quick-chat feels the same in text or voice.
    """
    data = request.get_json(silent=True) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"error": "Send JSON {\"text\": \"...\"}"}), 400

    raw_history = data.get("history") or []
    # Only trust role/content, and only the last 20 turns, to keep prompts bounded.
    history = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in raw_history if m.get("role") in ("user", "assistant")
    ][-20:]

    try:
        reply = svc.chat(svc.GENERIC_SYSTEM_PROMPT, history, user_text)
    except Exception as e:
        return jsonify({"error": f"LLM failed: {e}"}), 502

    return jsonify({"reply": reply})


_SENT_END = re.compile(r"(?<=[.!?])\s+")


@app.route("/api/call/<int:char_id>/turn_stream", methods=["POST"])
def call_turn_stream(char_id):
    """LIVE turn. STT -> streaming LLM -> streaming TTS, sentence by sentence.

    Returns a streamed body of raw PCM (s16le) audio chunks that begin
    arriving as soon as the first sentence is synthesised, so the client
    can start playing almost immediately. The full reply text is sent in
    the X-Reply-Text trailer header isn't reliable across servers, so the
    client should also call /history after playback to get the final text.
    """
    char = Character.query.get_or_404(char_id)
    if "audio" not in request.files:
        return jsonify({"error": "Upload audio as multipart field 'audio'"}), 400
    audio_bytes = request.files["audio"].read()
    if len(audio_bytes) < 1000:
        return jsonify({"error": "Audio too short (< 1KB)"}), 400

    try:
        user_text = svc.transcribe(audio_bytes, filename="audio.webm")
    except Exception as e:
        return jsonify({"error": f"STT failed: {e}"}), 502
    if not user_text or len(user_text) < 2:
        return jsonify({"error": "Didn't catch any speech", "user_text": user_text}), 200

    history = _call_history.setdefault(char_id, [])
    system_prompt = char.system_prompt()
    voice_id = char.voice_id

    @stream_with_context
    def generate():
        buffer = ""
        full_reply = []
        for token in svc.chat_stream(system_prompt, history, user_text):
            buffer += token
            # Flush whole sentences to TTS as soon as they complete.
            parts = _SENT_END.split(buffer)
            if len(parts) > 1:
                *complete, buffer = parts
                for sentence in complete:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    full_reply.append(sentence)
                    for chunk in svc.synthesize_stream(sentence, voice_id):
                        yield chunk
        # Flush any trailing text.
        tail = buffer.strip()
        if tail:
            full_reply.append(tail)
            for chunk in svc.synthesize_stream(tail, voice_id):
                yield chunk

        # Persist turn + roll memory (inside app context).
        reply_text = " ".join(full_reply).strip()
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply_text})
        if len(history) > 20:
            _call_history[char_id] = history[-20:]

    resp = Response(generate(), mimetype="audio/L16")
    resp.headers["X-User-Text"] = quote(user_text)
    resp.headers["X-Sample-Rate"] = str(Config.CARTESIA_SAMPLE_RATE)
    resp.headers["Access-Control-Expose-Headers"] = "X-User-Text, X-Sample-Rate"
    return resp


@app.route("/api/call/<int:char_id>/end", methods=["POST"])
def call_end(char_id):
    """Ends the session: summarise the conversation into the character's
    rolling memory so it's remembered next time, then clear in-memory history.
    """
    char = Character.query.get_or_404(char_id)
    history = _call_history.get(char_id, [])
    try:
        new_memory = svc.summarize_conversation(char.memory or "", history)
        char.memory = new_memory
        db.session.commit()
    except Exception as e:
        log.exception("memory summarise failed")
        return jsonify({"error": f"summarise failed: {e}"}), 502
    _call_history[char_id] = []
    return jsonify({"ok": True, "memory": char.memory})


@app.route("/api/call/<int:char_id>/history", methods=["GET"])
def call_history(char_id):
    Character.query.get_or_404(char_id)
    return jsonify({"history": _call_history.get(char_id, [])})


@app.route("/api/characters/<int:char_id>/transcript", methods=["GET"])
def get_transcript(char_id):
    """Raw turn-by-turn transcript of the character's most recent LIVE
    call (i.e. a real call made through voice_agent.py's WebSocket
    pipeline, saved automatically on hangup). Empty if no live call has
    happened yet. This is separate from /api/call/<id>/history, which
    only reflects Postman-driven /turn and /turn_stream test calls.
    """
    import json
    char = Character.query.get_or_404(char_id)
    try:
        transcript = json.loads(char.last_transcript) if char.last_transcript else []
    except (ValueError, TypeError):
        transcript = []
    return jsonify({"character_id": char_id, "transcript": transcript})


def _audio_with_text(audio: bytes, reply: str, user_text: str = ""):
    resp = send_file(BytesIO(audio), mimetype="audio/wav", download_name="reply.wav")
    resp.headers["X-Reply-Text"] = quote(reply)
    resp.headers["X-User-Text"] = quote(user_text)
    resp.headers["Access-Control-Expose-Headers"] = "X-Reply-Text, X-User-Text"
    return resp


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "path": request.path}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 25 MB)"}), 413


@app.errorhandler(500)
def server_error(e):
    log.exception("500: %s", e)
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


if __name__ == "__main__":
    print("\n  TalkToMyChild API on http://localhost:5000")
    print("  Health: http://localhost:5000/api/health\n")
    app.run(host="0.0.0.0", port=5000, debug=Config.DEBUG, threaded=True)
