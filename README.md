# TalkToMyChild

A child calls a loved one's voice — cloned with that person's consent — and
talks to them live, like a real phone call. The character remembers previous
conversations.

Two processes that share one SQLite database:

1. **Web app + REST API** (`app.py`) — the frontend (character gallery + call
   screen) and the API (character CRUD, voice cloning, previews).
2. **Live voice agent** (`voice_agent.py`) — the real-time call pipeline:
   mic → STT → LLM → cloned-voice TTS → speaker, with barge-in (interruption).

```
  Browser  ──HTTP──>  app.py (Flask)  ──>  index.html / call.html (frontend)
     │                    │
     │ WebSocket          │ shares
     ▼                    ▼
  voice_agent.py  ──>  instance/dadvoice.db  <──  app.py
     │
     └─uses─> tts_stt_llm.py ──> Groq (STT+LLM) + Cartesia (TTS+clone)
```

---

## File layout

```
.
├── app.py                              # Flask: frontend pages + REST API
├── voice_agent.py                      # Pipecat live-call agent (WebSocket)
├── config.py                           # Reads .env
├── models.py                           # User + Character (voice_id + memory)
├── tts_stt_llm.py                      # transcribe / chat / synthesize / clone / summarize
├── templates/
│   ├── index.html                      # Character gallery + create form
│   ├── call.html                       # Live call screen (voice + inline text chat)
│   └── quickchat.html                  # Quick Chat: no character, text or voice
├── static/uploads/                     # voice samples + images saved here
├── requirements.txt
├── .env.example
└── TalkToMyChild_postman_collection.json
```

**You don't place the HTML anywhere yourself** — the frontend lives in
`templates/` and Flask serves it. Just open `http://localhost:5000`.

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
```

Fill in `.env`:

- `GROQ_API_KEY` — https://console.groq.com  (STT + LLM)
- `CARTESIA_API_KEY` — https://play.cartesia.ai → Settings → API Keys (TTS + cloning)

No ngrok / `PUBLIC_BASE_URL` needed — Cartesia clones a voice once and
references it by id, so your server never has to be publicly reachable.

---

## Run it — two terminals

### Terminal 1 — the web app

```bash
python app.py
# → http://localhost:5000
```

Open **http://localhost:5000** in your browser. You'll see the character
gallery. (Health/debug: http://localhost:5000/api/health — `cloning_enabled`
should be `true`.)

### Create a character (in the browser)

Click **Create character**, fill in the name, add a voice sample (5–20s of the
person's own speech, **with their consent**), optionally an image, and save.
The voice is cloned via Cartesia and a `voice_id` is stored. Note the character
you just made — you'll start an agent for it next.

> You can also create characters via Postman (import the collection) or curl —
> see "REST API" below.

### Terminal 2 — the live voice agent

Each character you want to call needs an agent running. Find the character id
(visible in the URL when you click a character, e.g. `/call?char=1`), then:

```bash
python voice_agent.py --char 1 --port 8765
```

You should see `Voice agent for 'Dad' live on ws://0.0.0.0:8765`.

### Make the call

Back in the browser, click the character (or go to
`http://localhost:5000/call?char=1`). Tap **Start call**, allow the microphone,
and talk. The character greets you, listens (no button — it detects when you
stop), replies in the cloned voice, and you can talk over it to interrupt.

When you hang up, the conversation is summarized into the character's long-term
memory. Next call, it remembers — you'll see "Remembers your last call" on the
character card.

---

## How memory works

- **During a call:** Pipecat keeps the conversation in context.
- **On hangup:** `voice_agent.py` merges the call into the character's memory
  via `summarize_conversation()` and saves it to the DB.
- **Next call:** that memory is injected into the system prompt, so the
  character remembers the child's name, pets, what they talked about.

One rolling summary per character — simple, good for the MVP. Later you can
split it into stable facts + recent episode summaries for better recall.

---

## Transcripts

Every real live call (through `voice_agent.py`) saves its raw turn-by-turn
transcript to the character, separate from the AI-summarised `memory`:

- `GET /api/characters/<id>/transcript` — raw `{"role", "content"}` turns
  from the character's most recent live call. Overwritten each call.
- The character gallery shows a **Transcript** button on any card that has
  one; the call screen shows a **View transcript** button after you hang up.
- `GET /api/call/<id>/history` is different: it only reflects Postman/curl
  turns made through `/turn` or `/turn_stream`, not real live calls.

---

## Text chat (no voice)

Every call screen has a **Type instead** link so you can chat by typing
instead of talking — handy for testing, or for a quiet conversation. It uses
`POST /api/call/<id>/message` (JSON `{"text": "..."}` → `{"reply": "..."}`),
sharing the same conversation history as voice turns, so switching between
typing and talking mid-conversation works.

---

## Quick Chat — no character

Sometimes you don't want a specific character, just a plain assistant to
talk or type with. Open **`/quickchat`** (also linked from the home page):

- **Text**: type directly on the page, via the stateless
  `POST /api/chat/generic` endpoint (no character, nothing saved).
- **Voice**: tap "🎙️ Voice" to jump into a live call with no character,
  same as `/call?generic=1`.

The generic voice call needs its own agent process, since it isn't tied to
any character id:

```bash
python voice_agent.py --generic --port 8766
```

Or start it together with your character agent(s) via `main.py`:

```bash
python main.py --char 1 --generic-port 8766
```

Nothing from Quick Chat is saved anywhere — no memory, no transcript. It
resets every time you reload the page.

---

## REST API (for Postman / curl testing)

Import `TalkToMyChild_postman_collection.json`, set `baseUrl` to
`http://localhost:5000`, run top-to-bottom.

| Endpoint                            | Purpose                              |
|-------------------------------------|--------------------------------------|
| `GET /api/health`                   | Keys loaded, agent ws url            |
| `POST /api/test/stt`                | Groq Whisper (upload audio)          |
| `POST /api/test/llm`                | Groq Llama                           |
| `POST /api/test/tts`                | Cartesia TTS (preset or `voice_id`)  |
| `GET/POST /api/characters`          | List / create characters             |
| `POST /api/characters/<id>/preview` | Hear the cloned voice                |
| `POST /api/call/<id>/turn`          | Non-streaming full turn (easy to test)|
| `POST /api/call/<id>/message`       | Text-only turn (no audio in/out)     |
| `POST /api/chat/generic`            | Text-only chat, no character (stateless) |
| `GET /api/characters/<id>/transcript` | Raw transcript of last live call   |
| `POST /api/call/<id>/end`           | Summarize → save memory              |

The live agent and streaming endpoints can't be "heard" in Postman — use the
call screen.

---

## Notes & limits (MVP)

- **One agent per call, launched by hand.** In production, have the backend
  spawn an agent process per call on a free port and hand its `ws://` URL to the
  client (the call screen already reads `agent_ws_url` from `/api/health`, or
  accepts `?ws=` override).
- **Microphone needs a secure context.** localhost is fine. For a phone or a
  remote IP you need HTTPS or the mic won't start.
- **State** (`_call_history`) is in-memory — move to Redis for multi-user.
- **DB** is SQLite — use Postgres for production.
- **Consent + safety.** Only clone voices from consenting people. The gallery
  should hold family / properly licensed voices, never scraped celebrity or
  copyrighted-character voices.

---

## Troubleshooting

**`GET / 404`** — fixed: `/` now serves the gallery. If you still see it, you're
running an old `app.py`.

**Mic won't start** — you're on a non-secure address. Use `localhost` or HTTPS.

**"Couldn't reach the call"** — the voice agent isn't running for that
character, or the port/ws URL is wrong. Start `python voice_agent.py --char <id>`.

**Cloned voice sounds like a preset** — cloning failed at creation (agent falls
back to the preset id). Check the create response had a non-null `voice_id`;
re-test with `POST /api/test/tts` and that `voice_id`.

**Character doesn't remember** — memory saves on hangup. Ensure the call fully
disconnects (agent logs `saving memory`); a hard kill skips the summary.
