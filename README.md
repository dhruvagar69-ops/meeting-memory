# Meeting Memory

Ask your meetings questions and get answers that point to the exact speaker and moment.

Built for the DEV **Hacktoberfest Weekend Challenge: "Build for a Friend"**. Started 2026-10-02 (first commit 2026-10-03).

Meeting Memory loads meeting transcripts, pulls out decisions, action items and open questions (each tied to a quote), and lets you ask questions across all the meetings. Every answer cites the lines it came from. If the meetings do not contain the answer, it says "Not found" instead of guessing.

**Who it is for:** someone who sits in recurring project meetings and loses track of what was decided. It was built and evaluated on the public AMI Meeting Corpus. It was **not** tested with a real friend or team.

## How it works

```
transcript (AMI or plain text)
   -> parse into segments: speaker, start time, text, id like ES2008b-s0216
   -> SQLite (segments + a keyword index)

EXTRACT  (per meeting, in chunks)
   model proposes decisions / actions / questions with a quote and a segment id
   -> dropped if the segment does not exist, or the quote is not word-for-word in it
   -> owner must be a real speaker (otherwise "Unassigned")
   -> deadline kept only if it is actually spoken
   -> stored in SQLite

ASK  (per question, across all meetings)
   keyword search (BM25) -> best matches plus neighbouring lines, merged into windows
   -> nothing found?  "Not found in these meetings." (the model is not called)
   -> model answers from the numbered windows only and cites them like [S1]
   -> citations to windows that were not provided are removed
   -> an answer with no valid citation is flagged UNVERIFIED
```

Retrieval is plain keyword search (SQLite FTS5 with BM25 and simple prefix stemming). There are **no embeddings**. Two optional add-ons are available:

- `--expand`: the model first suggests extra search words (synonyms) for the question. This costs one more model call.
- `--per-meeting N`: also takes the N best matches from every meeting, so one noisy meeting does not crowd out the others.

## Models and privacy

Transcripts and the database stay on your machine. The model backend is configurable:

- **Local:** Ollama (`--backend ollama`, the default). Nothing leaves your machine.
- **Hosted open-weight models:** Groq or OpenRouter (`--backend groq` or `--backend openrouter`). The transcript text is sent to that provider.
- **Any OpenAI-compatible server**, such as LM Studio: `--backend openai --base-url http://localhost:1234/v1`.

**All results in this README were produced with open-weight models hosted on Groq, not locally.**

- Extraction: `openai/gpt-oss-120b` (reasoning effort medium).
- Question answering: `qwen/qwen3.8-27b`. The free-tier daily token limit for the 120b model had been used up by the extraction runs, so the question evaluation used this model.

The local path is implemented and covered by tests, but I did not run the evaluation with it.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q

python -m ingest.download_ami                       # ES2008a-d transcripts only
python -m ingest.parse data/ami ES2008a             # eyeball the parsed output first
python -m ingest.add_meeting ami data/ami ES2008a   # repeat for ES2008b, ES2008c, ES2008d
```

Set your API key first and never commit it:

```
# PowerShell
$env:GROQ_API_KEY = "your_key_here"
# Windows cmd
set GROQ_API_KEY=your_key_here
# Mac / Linux
export GROQ_API_KEY=your_key_here
```

List the models your key can use, since names change over time:

```
python -m extract.extract --backend groq --list-models
```

Extract decisions, actions and questions:

```
python -m extract.extract ES2008b --backend groq --model openai/gpt-oss-120b --reasoning-effort medium --max-tokens 3000 --show-dropped
```

Ask a question:

```
python -m ask.ask "What is the project manager's name?" --backend groq --model qwen/qwen3.8-27b
```

Example output:

```
The project manager's name is Rose Lindgren [S3].

Sources:
[S3] ES2008a 00:31-01:01
    A (PM): Okay. Good morning everybody. ... My name is Rose Lindgren. I I'll be the Project Manager.
    ...
```

Add `--expand` and/or `--per-meeting 3 --max-windows 8` to try the retrieval add-ons.

A question the meetings cannot answer:

```
python -m ask.ask "What is the capital of France?" --backend groq --model qwen/qwen3.8-27b
Not found in these meetings.
```

A local model works the same way, without `--backend groq`: `python -m extract.extract ES2008b --model <ollama-model>`.

Plain-text transcripts (for your own meetings) look like this, one utterance per line:

```
[00:01:23] Riya: let's go with the main hall
Aman: sounds good
```

Load one with `python -m ingest.add_meeting plain path/to/meeting.txt --id my-meeting`.

## Evaluate

```
# run the 20 questions in eval/questions.json
python -m eval.run_eval --tag plain --backend groq --model qwen/qwen3.8-27b --max-tokens 2000
python -m eval.run_eval --tag combined --expand --per-meeting 3 --max-windows 8 --backend groq --model qwen/qwen3.8-27b --max-tokens 2000
# if a run stops (for example a rate limit), repeat the same command with --resume

# fill the `correct` column by hand (yes / partial / no), then score
python -m eval.run_eval --score eval/results_qwen.csv

# export extracted items for hand-labelling, then score
python -m eval.export_items ES2008d --out eval/d_120b.csv
python -m eval.export_items --score eval/d_120b.csv
```

Runs print `tokens used by this run`. A 20-question evaluation used roughly 22,000 to 23,000 tokens.

## Results

Everything below was labelled by hand on the AMI ES2008 meeting series. The samples are small, so treat the numbers as indicative only.

### Question answering

20 questions: 16 answerable (6 of them need more than one meeting) and 4 the meetings should not answer.

**Hand-labelled answer correctness, 16 answerable questions**

| Setup | Yes | Partial | No | Strict | Lenient |
|---|---|---|---|---|---|
| Plain keyword search | 7 | 5 | 4 | 0.44 | 0.75 |
| Synonym expansion + per-meeting sampling | 7 | 7 | 2 | 0.44 | 0.88 |

Strict counts only "yes". Lenient counts "yes" and "partial". Counting all 20 questions, strict is 0.55 for both and lenient is 0.80 versus 0.90.

All 4 unanswerable questions returned "Not found" in every setup. This is weaker evidence than it looks: for questions with no matching keywords, the model is never called, so it is not a test of the model's own refusal. I chose these questions as ones the meetings should not answer.

**Automatic retrieval metrics, 16 answerable questions**

These check whether a specific segment I wrote down as the expected source was retrieved or cited. The model can give a good answer from a different line, so these numbers understate quality. Only the first and last rows were hand-labelled for correctness.

| Setup | Expected line retrieved | Expected line cited | Cross-meeting questions cited (of 6) | Answerable but "Not found" |
|---|---|---|---|---|
| Plain | 7/16 | 7/16 | 1 | 3 |
| `--expand` | 8/16 | 8/16 | 3 | 2 |
| `--per-meeting 3` (8 windows) | 10/16 | 9/16 | 1 | 3 |
| Both | 9/16 | 8/16 | 3 | 2 |

Findings:
- Retrieval, not the model, is the main weakness. When the expected line was retrieved, it was cited.
- Each add-on helped a little, in different places, and combining them did not stack. With 16 questions, a difference of one or two rows is noise, so I do not rank the variants.
- Zero answers came back with no valid citation.

**A failure worth keeping:** asked what was decided about the price, the system found the 12.5 Euro production target but missed the decision to sell the remote separately at 25 Euro, because that line says "Euro" and never "price". Keyword search cannot connect the two words.

### Extraction

Measured on one meeting (ES2008d), 44 items extracted with `openai/gpt-oss-120b`.

| Type | Items | Supported | Partial | Unsupported | Strict precision | Lenient precision |
|---|---|---|---|---|---|---|
| Decision | 10 | 7 | 3 | 0 | 0.70 | 1.00 |
| Action | 7 | 3 | 1 | 3 | 0.43 | 0.57 |
| Question | 27 | 5 | 2 | 20 | 0.19 | 0.26 |
| All | 44 | 15 | 6 | 23 | 0.34 | 0.48 |

- Decisions are the strong result. Open questions are weak: most "questions" were answered in the next few lines.
- The verification step dropped items whose quote was not in the cited segment. In the extraction runs, it caught fabricated items such as a "decision" whose quote did not exist.
- Extraction **recall was not measured**.

Model size mattered a great deal on this meeting, although the settings differed in more than one way, so it is not a clean model comparison:

| Setup | Items kept from ES2008d |
|---|---|
| `gpt-oss-20b`, low effort, default chunks | 3 |
| `gpt-oss-20b`, low effort, 5,000-character chunks | 7 |
| `gpt-oss-20b`, medium effort | failed (HTTP 413, token limit), no result |
| `gpt-oss-120b`, medium effort | 44 |

## Known limits

- Only 20 questions and one meeting series (AMI ES2008). Differences of one or two rows are noise.
- **The 20 questions and their expected answers were written by an AI assistant**, using lines that had appeared in earlier extraction and parser output. This biases them toward content the extractor had already found. I did not read the full transcripts before writing them.
- Verdicts started from AI-suggested labels and were then reviewed against the transcripts.
- One labeller, no inter-rater check.
- Retrieval is keyword-only and misses some answers entirely, for example which ideas the team left out.
- The extractor often reports questions that someone answers in the next line, and sometimes presents suggestions or finished tasks as decisions or open actions.
- Extraction precision was measured on one meeting; recall was not measured.
- The hosted models mean transcript text left the machine in these runs. Use the local backend for private meetings.
- No real friend or team tested it, so there is no user feedback.

## Repository layout

```
ingest/    download_ami.py, parse.py, add_meeting.py   (AMI and plain-text parsing, loading)
store/     schema.sql, db.py                            (SQLite: meetings, segments, items, keyword index)
extract/   prompts.py, extract.py, verify.py, llm.py    (extraction, checks, model backends)
ask/       retrieve.py, ask.py                          (retrieval, cited answers)
eval/      questions.json, run_eval.py, export_items.py, results and label CSVs
tests/     unit tests (run with: python -m pytest -q)
```

## Data and attribution

- **AMI Meeting Corpus** (CC BY 4.0): Carletta et al., "The AMI Meeting Corpus: A Pre-announcement", 2005. https://groups.inf.ed.ac.uk/ami/corpus/
- The data is downloaded by script and git-ignored. Nothing from AMI is redistributed in this repository.
- Names in the transcripts (for example "Rose Lindgren") come from the AMI scenario recordings.

## Notes

- The parser tests use a hand-made fixture that mimics the AMI annotation format. The parser was also run on the real downloaded ES2008a files and the output was checked. Always run `python -m ingest.parse data/ami <meeting>` on a real download and read the result.
- Free-tier hosted models have daily and per-minute token limits. A daily limit stops a run immediately. Pick another `--model`, wait for the limit to refill, or resume with `--resume`.
- Commits after the challenge deadline (Mon 2026-10-05 12:29 PM IST): none at the time of writing.
- Built with the help of Claude (code, tests and evaluation design). I made the project decisions and did the labelling.

## License

MIT. See `LICENSE`.
