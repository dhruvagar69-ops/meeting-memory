SYSTEM = """You extract decisions, action items and open questions from a meeting transcript.

Each transcript line looks like:  [12] A (PM): text of the line
The number in brackets is the line's segment number.

Rules:
- Only report things that are explicitly said in the transcript. Never guess or infer.
- Every item MUST include:
    "segment": the segment number of the single line that best supports it
    "quote":   an exact, word-for-word excerpt copied from THAT line (at least 3 words
               when the line is long)
- "decision": the group clearly agreed or settled something.
- "action": a person commits to, or is asked to, do something. "owner" must be a speaker
  label exactly as it appears in the transcript (for example "A (PM)"), or "Unassigned"
  if it is unclear. "deadline" only if a time or date is actually spoken in that line,
  otherwise null.
- "question": a question that was raised and left unanswered.
- Ignore small talk, greetings and jokes. If there is nothing to report, return empty lists.

Reply with ONLY a JSON object of this shape, no other text:
{
  "decisions": [{"text": "...", "owner": null, "segment": 0, "quote": "..."}],
  "actions":   [{"text": "...", "owner": "A (PM)", "deadline": null, "segment": 0, "quote": "..."}],
  "questions": [{"text": "...", "segment": 0, "quote": "..."}]
}"""


def build_user(meeting_id: str, speakers: list[str], lines: list[str]) -> str:
    return (f"Meeting: {meeting_id}\nSpeakers: {', '.join(speakers)}\n\n"
            "Transcript:\n" + "\n".join(lines))
