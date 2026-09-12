"""DB models. Single-user demo: one User, many Characters."""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), default="Demo")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    characters = db.relationship("Character", backref="user",
                                 lazy=True, cascade="all, delete-orphan")


class Character(db.Model):
    __tablename__ = "characters"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.String(20))
    gender = db.Column(db.String(20))
    role = db.Column(db.String(80))
    category = db.Column(db.String(40))
    description = db.Column(db.Text)

    image_path = db.Column(db.String(255))         # relative to /static
    voice_sample_path = db.Column(db.String(255))  # the uploaded clip we cloned from

    # Cartesia voice id returned by clone_voice() — set once at creation.
    # If null, synthesis falls back to the preset voice.
    voice_id = db.Column(db.String(120))

    # Rolling long-term memory: a short notes blob updated after each call.
    # Injected into the system prompt so the character "remembers".
    memory = db.Column(db.Text, default="")

    # Raw transcript of the most recent LIVE call (from voice_agent.py),
    # stored as a JSON string list of {"role": ..., "content": ...}.
    # Overwritten each call — not accumulated. Separate from `memory`,
    # which is the AI-summarised version used for long-term recall.
    last_transcript = db.Column(db.Text, default="")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "role": self.role,
            "category": self.category,
            "description": self.description,
            "image_path": self.image_path,
            "voice_sample_path": self.voice_sample_path,
            "voice_id": self.voice_id,
            "has_memory": bool(self.memory),
            "has_transcript": bool(self.last_transcript),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def system_prompt(self) -> str:
        bits = [f"You are {self.name}, speaking on a phone call with a child."]
        if self.role:        bits.append(f"Your role: {self.role}.")
        if self.age:         bits.append(f"Your age: {self.age}.")
        if self.gender:      bits.append(f"Your gender: {self.gender}.")
        if self.description: bits.append(f"About you: {self.description}")

        if self.memory:
            bits.append(
                "Here is what you remember about this child from previous "
                f"conversations: {self.memory}"
            )

        bits.append(
            "IMPORTANT — stay grounded: only use the facts explicitly given "
            "above (your name, role, age, gender, description, and memory). "
            "Do NOT invent new personal details, backstory, events, jobs, "
            "relatives, pets, places, or anything about yourself or the "
            "child that was not explicitly stated. If asked something you "
            "don't know or wasn't given to you, respond naturally in "
            "character without making up specifics — you can deflect "
            "warmly or ask the child instead of inventing an answer."
        )

        bits.append(
            "Speak warmly, in character, in first person. "
            "Keep replies SHORT — 1 to 3 sentences. "
            "Never reveal you are an AI. Never break character. "
            "Avoid scary, violent, sexual, or age-inappropriate content. "
            "If the child seems upset or in danger, gently suggest a trusted grown-up. "
            "No emojis, no markdown, no stage directions — only spoken words."
        )
        return " ".join(bits)
