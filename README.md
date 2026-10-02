# Meeting Memory

A local tool that remembers what everyone said across meetings and answers
questions with the exact speaker and timestamp. Built for the DEV Hacktoberfest
Weekend Challenge ("Build for a Friend"). Started 2026-10-02.

Transcripts and the database stay on your machine. The model backend is configurable:
local Ollama (nothing leaves the machine) or a hosted open-weight model via Groq/OpenRouter
(transcript text is sent to that provider). The write-up must say which backend produced each result.

## Status (step 1 of the build)
- [x] Repo layout, AMI download script, transcript parsers, SQLite schema
- [ ] Extraction with citation checks (`extract/`)
- [ ] Embeddings + hybrid retrieval, ask loop (`ask/`)
- [ ] Evaluation (`eval/`), UI, write-up

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m ingest.download_ami                      # ES2008a-d transcripts only
python -m ingest.parse data/ami ES2008a            # eyeball the parsed output first
python -m ingest.add_meeting ami data/ami ES2008a  # repeat for b, c, d
python -m ingest.add_meeting plain my_meeting.txt --id club-01 --title "Club planning (synthetic)"
python -m pytest -q

# extraction, hosted open-weight model (set the key first, never commit it)
#   Windows cmd: set GROQ_API_KEY=...      Mac/Linux: export GROQ_API_KEY=...
python -m extract.extract ES2008b --backend groq --model <model-name> --show-dropped
# or local: python -m extract.extract ES2008b --model <ollama-model> --show-dropped
```

Plain-text transcripts look like:
```
[00:01:23] Riya: let's go with the main hall
Aman: sounds good
```

## Data and attribution
- **AMI Meeting Corpus** (CC BY 4.0): Carletta et al., "The AMI Meeting Corpus: A Pre-announcement", 2005.
  https://groups.inf.ed.ac.uk/ami/corpus/
- Data is downloaded by script and git-ignored; nothing from AMI is redistributed here.
- Self-written meetings are synthetic and labelled as such.

## Notes
- The parser tests use a hand-made fixture that mimics the AMI NXT format, not real AMI files.
  Always run `python -m ingest.parse` on the real download and check it.
- Any commits after the challenge deadline (Mon 2026-10-05 12:29 PM IST) must be listed here.
