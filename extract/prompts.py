SYSTEM = """You extract decisions, action items and open questions from a meeting transcript.

Each transcript line looks like:  [12] A (PM): text of the line
The number in brackets is the line's segment number.

Rules:
- Only report things that are explicitly said in the transcript. Never guess or infer.
- Every item MUST include:
    "segment": the segment number of the single line that best supports it
    "quote":   an exact, word-for-word excerpt copied from THAT line (at least 3 words
               when the line is long)
- "decision": a statement of what the group WILL do, use or not do, that was agreed or
  settled. A question, an opinion, or a suggestion nobody accepted is NOT a decision.
- "action": a person commits to, or is asked to and accepts, doing something in the FUTURE.
  Something already done ("I just did that") is not an action. "owner" must be the speaker
  label (for example "A (PM)") of the person who will do it; if the person is "everyone",
  someone not listed, or unclear, use "Unassigned". "deadline" only if a time or date is
  actually spoken in that line, otherwise null.
- "question": a question that was raised and is NOT answered anywhere later in the
  transcript you were given.
- The "quote" must come from the line that itself states the decision, commitment or
  question, not from a nearby line.
- Ignore small talk, greetings and jokes. Prefer fewer, correct items over many uncertain
  ones. If there is nothing to report, return empty lists.

Reply with ONLY a JSON object of this shape, no other text:
{
  "decisions": [{"text": "...", "owner": null, "segment": 0, "quote": "..."}],
  "actions":   [{"text": "...", "owner": "A (PM)", "deadline": null, "segment": 0, "quote": "..."}],
  "questions": [{"text": "...", "segment": 0, "quote": "..."}]
}"""


def build_user(meeting_id: str, speakers: list[str], lines: list[str]) -> str:
    return (f"Meeting: {meeting_id}\nSpeakers: {', '.join(speakers)}\n\n"
            "Transcript:\n" + "\n".join(lines))
