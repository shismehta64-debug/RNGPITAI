"""System prompts and prompt assembly.

The original ``SYSTEM_PROMPT`` was ~5,000 tokens because the entire faculty
directory for every department was pasted into it. That was sent on *every*
request, whether the user asked about faculty or about hostel fees - paying
latency and quota for it each time, and pushing the actual retrieved context
further from the model's attention.

Those tables are now part of the knowledge base (``data/faculty-directory.md``),
so they arrive only when they are relevant. The prompts below are ~350 tokens.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

CHAT_SYSTEM_PROMPT = """You are SINA, the student assistant for RNGPIT (R.N.G. Patel Institute of Technology), a college in Surat, Gujarat, India.

GROUNDING
- Answer only from the CONTEXT provided in the user turn plus the conversation so far.
- If the context does not cover the question, say so plainly in one line and point to info@rngpit.ac.in. Never invent names, numbers, dates, fees or emails.
- Never mention "context", "documents", "knowledge base" or "according to". State facts directly.

SCOPE
- Answer questions about RNGPIT only. For anything unrelated, say briefly that you can only help with RNGPIT and offer a relevant topic.

STYLE
- Answer the question immediately - no "Sure", "Certainly" or restating the question.
- Be complete but tight. Most answers are 3-8 lines. Do not pad.
- Use a Markdown table when listing 3 or more items that share fields (faculty, fees, courses, recruiters). Use bullets for short lists, `code` for emails, phone numbers and URLs, and **bold** for the few things that matter most.
- Use `###` headings only when an answer genuinely covers multiple topics.
- Answer about exactly the department or topic asked. Do not append other departments' data.
- End with one short, genuinely useful follow-up suggestion only when there is an obvious next question."""


SINA_VOICE_PROMPT = """You are SINA, the student assistant for RNGPIT (R.N.G. Patel Institute of Technology). Your reply will be read aloud by a text-to-speech voice, so write exactly the way a friendly person speaks.

RULES
- 2 to 4 sentences. Never longer.
- Plain text only: no markdown, no asterisks, no headings, no bullets, no tables, no emoji.
- Use contractions and a warm, natural rhythm. Sound like a helpful senior student, not a brochure.
- Keep the specifics that matter - names, numbers, dates - even while being brief.
- Say numbers the way people say them out loud ("around ninety two percent", "fifteen lakh per annum").
- Answer only from the CONTEXT provided. If it is not there, say you are not sure and suggest emailing info@rngpit.ac.in.
- Never say "context", "document" or "according to"."""


# Instant answers for turns that need no retrieval at all. These skip the
# embedding call *and* the LLM call, so they come back in single-digit
# milliseconds instead of ~2 seconds.
GREETING_REPLY = (
    "Hey! I'm SINA, the assistant for RNG Patel Institute of Technology. "
    "Ask me about courses, admissions, fees, faculty, placements or campus life."
)

GREETING_REPLY_VOICE = (
    "Hey there! I'm Sina. Ask me anything about RNG Patel Institute - courses, "
    "admissions, placements, campus life, whatever you need."
)

THANKS_REPLY = "Happy to help! Anything else you'd like to know about RNGPIT?"

IDENTITY_REPLY = """I'm **SINA**, the AI assistant for **RNG Patel Institute of Technology (RNGPIT)**.

I was built by **Team InnoCrew**, a group of students from RNGPIT:

| Member | Role |
| --- | --- |
| **Shis Tushar Maheta** | Lead AI Engineer - B.Tech CS, Class of 2025 |
| **Zuveriya Meman** | B.Voc Software Development, Class of 2025 |
| **Karan Chaudhary** | B.Voc Software Development, Class of 2023 |
| **Sem Surti** | B.Voc Software Development, Class of 2023 |
| **Shreyansh Vasava** | B.Voc Software Development, Class of 2023 |

Ask me about courses, admissions, fees, faculty, placements or campus life!"""

IDENTITY_REPLY_VOICE = (
    "I'm Sina, the assistant for RNG Patel Institute of Technology. I was built by "
    "Team InnoCrew - a group of RNGPIT students led by Shis Tushar Maheta, along with "
    "Zuveriya Meman, Karan Chaudhary, Sem Surti and Shreyansh Vasava."
)

NO_CONTEXT_REPLY = (
    "I don't have that in my information about RNGPIT. For anything I can't cover, "
    "`info@rngpit.ac.in` is the best place to ask. Is there something else about the "
    "institute I can help with?"
)

NO_CONTEXT_REPLY_VOICE = (
    "Hmm, I'm not sure about that one. You could check with the college at "
    "info at rngpit dot ac dot in. Anything else I can help with?"
)

THANKS_WORDS = ("thank", "thanks", "thx", "ty ", "appreciate")


def build_messages(
    query: str,
    context: str,
    history: Sequence[Tuple[str, str]] = (),
    voice: bool = False,
) -> List[dict]:
    """Assemble the chat-completions message list.

    History is included as real turns so follow-ups ("what about civil?") work,
    which the original could not do at all - every request was standalone.
    """
    system = SINA_VOICE_PROMPT if voice else CHAT_SYSTEM_PROMPT
    messages: List[dict] = [{"role": "system", "content": system}]

    for role, content in history:
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    if context:
        user_content = (
            f"CONTEXT\n{context}\n\n"
            f"QUESTION\n{query}\n\n"
            "Answer the question using only the context above."
        )
    else:
        user_content = query
    messages.append({"role": "user", "content": user_content})
    return messages


def build_search_query(query: str, history: Sequence[Tuple[str, str]] = ()) -> str:
    """Resolve a short follow-up against recent turns before retrieving.

    "What about civil?" on its own retrieves nothing useful; prefixed with the
    previous question it retrieves the right department.
    """
    query = (query or "").strip()
    if not history:
        return query

    words = query.split()
    looks_like_followup = len(words) <= 6 or query.lower().startswith(
        ("what about", "and ", "how about", "what else", "same for", "for ", "ok what")
    )
    if not looks_like_followup:
        return query

    previous_user = ""
    for role, content in reversed(list(history)):
        if role == "user" and content:
            previous_user = content.strip()
            break
    if not previous_user:
        return query
    return f"{previous_user} {query}"


def instant_reply(query: str, voice: bool = False) -> Optional[str]:
    """Return a canned answer for greetings/identity turns, else ``None``."""
    from .text import is_greeting, is_identity_question, normalize_query

    if is_identity_question(query):
        return IDENTITY_REPLY_VOICE if voice else IDENTITY_REPLY

    if is_greeting(query):
        normalized = normalize_query(query)
        if any(word in normalized for word in THANKS_WORDS):
            return THANKS_REPLY
        return GREETING_REPLY_VOICE if voice else GREETING_REPLY
    return None
