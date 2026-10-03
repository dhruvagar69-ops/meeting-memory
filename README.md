# Meeting Memory

A tool that remembers what everyone said across meetings and answers questions with the exact speaker and timestamp. Built for the DEV Hacktoberfest Weekend Challenge ("Build for a Friend"). Started 2026-10-02.

Transcripts and the database stay on your machine. The model backend is configurable: local Ollama (nothing leaves the machine) or a hosted open-weight model via Groq/OpenRouter (transcript text is sent to that provider). Results below were produced with open-weight models on Groq: `qwen/qwen3.8-27b` for questions and `openai/gpt-oss-120b` for extraction.

## What it does

- **Extract:** pulls decisions, actions and questions from a meeting, each with a quote and a citation check (`extract/`).
- **Ask:** hybrid retrieval over all meetings, then an answer with speaker and timestamp citations. If the meetings don't contain the answer, it says "Not found" (`ask/`).
- **Evaluate:** scoring scripts for answer accuracy and extraction precision (`eval/`).

## Quick start

```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m ingest.download_ami                      # ES2008a-d transcripts only
python -m ingest.parse data/ami ES2008a            # eyeball the parsed output first
python -m ingest.add_meeting ami data/ami ES2008a  # repeat for b, c, d
python -m pytest -q
```

Set your key first and never commit it:
Windows cmd `set GROQ_API_KEY=...`, Mac/Linux `export GROQ_API_KEY=...`

```
# ask a question
python -m ask.ask "What is the project manager's name?" --backend groq --model qwen/qwen3.8-27b

# extract decisions, actions and questions
python -m extract.extract ES2008b --backend groq --model openai/gpt-oss-120b --reasoning-effort medium --max-tokens 3000

# score hand-labelled results
python -m eval.run_eval --score eval/results_qwen.csv
python -m eval.export_items --score eval/d_120b.csv
```

A local model also works: `python -m extract.extract ES2008b --model <ollama-model>`.

Plain-text transcripts look like:

```
[00:01:23] Riya: let's go with the main hall
Aman: sounds good
```

## Results

Hand-labelled by me on the AMI ES2008 meetings. Small samples, so treat them as indicative only.

**Question answering (20 questions, 4 unanswerable)**

| Setup | yes / partial / no | Strict | Lenient |
|---|---|---|---|
| results_qwen | 11 / 5 / 4 | 0.55 | 0.80 |
| results_qwen_combined | 11 / 7 / 2 | 0.55 | 0.90 |

All 4 unanswerable questions correctly returned "Not found" in both setups. On the 16 answerable questions only, strict is 0.44 for both and lenient is 0.75 vs 0.88.

**Extraction (44 items from the last meeting)**

| Type | Items | Strict precision | Lenient precision |
|---|---|---|---|
| Decision | 10 | 0.70 | 1.00 |
| Action | 7 | 0.43 | 0.57 |
| Question | 27 | 0.19 | 0.26 |
| All | 44 | 0.34 | 0.48 |

## Known limits

- Retrieval misses some answers entirely, for example which ideas the team left out, in both setups.
- The extractor often reports questions that someone answers in the next line.
- Only 20 questions and one meeting series, so differences of one or two rows are noise.
- Verdicts started from AI-suggested labels and were then reviewed against the transcripts.

## Data and attribution

- AMI Meeting Corpus (CC BY 4.0): Carletta et al., "The AMI Meeting Corpus: A Pre-announcement", 2005. https://groups.inf.ed.ac.uk/ami/corpus/
- Data is downloaded by script and git-ignored; nothing from AMI is redistributed here.
- Self-written meetings are synthetic and labelled as such.

## Notes

- The parser tests use a hand-made fixture that mimics the AMI NXT format, not real AMI files. Always run `python -m ingest.parse` on the real download and check it.
- Any commits after the challenge deadline (Mon 2026-10-05 12:29 PM IST) are listed here.