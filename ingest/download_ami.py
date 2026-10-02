"""Download the AMI public manual annotations and keep only the transcript files
for the meetings you ask for.

    python -m ingest.download_ami                          # ES2008a-d
    python -m ingest.download_ami --meetings ES2002a ES2002b

Output layout (what ingest.parse expects):
    data/ami/words/<meeting>.<speaker>.words.xml
    data/ami/segments/<meeting>.<speaker>.segments.xml
    data/ami/corpusResources/meetings.xml

AMI Meeting Corpus: signals and transcripts are CC BY 4.0.
Credit: Carletta et al., "The AMI Meeting Corpus: A Pre-announcement", 2005.
https://groups.inf.ed.ac.uk/ami/corpus/
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ZIP_NAME = "ami_public_manual_1.6.2.zip"
URLS = [
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/" + ZIP_NAME,
    # Mirror of the same official archive (CC BY 4.0), used as a fallback.
    "https://huggingface.co/datasets/FluidInference/ami-corpus-mirror/resolve/main/annotations/" + ZIP_NAME,
]
DEFAULT_MEETINGS = ["ES2008a", "ES2008b", "ES2008c", "ES2008d"]


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        print(f"Downloading {url}")
        tmp = dest.with_suffix(".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "meeting-memory/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            tmp.rename(dest)
            return
        except Exception as exc:  # noqa: BLE001 - fall through to the next source
            print(f"  failed: {exc}")
            tmp.unlink(missing_ok=True)
    sys.exit("Could not download the AMI annotations. Get the zip manually from "
             f"https://groups.inf.ed.ac.uk/ami/download/ and save it as {dest}")


def extract(zip_path: Path, out_dir: Path, meetings: list[str]) -> dict[str, int]:
    """Extract words/segments files for `meetings` plus meetings.xml.
    Returns {meeting_id: number of files extracted} for meetings that were found."""
    out_dir = Path(out_dir)
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            p = PurePosixPath(name)
            parent = p.parts[-2] if len(p.parts) > 1 else ""
            if parent in ("words", "segments") and p.name.endswith(f".{parent}.xml"):
                meeting = p.name.split(".")[0]
                if meeting in meetings:
                    dest = out_dir / parent / p.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))
                    counts[meeting] = counts.get(meeting, 0) + 1
            elif parent == "corpusResources" and p.name == "meetings.xml":
                dest = out_dir / "corpusResources" / "meetings.xml"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meetings", nargs="+", default=DEFAULT_MEETINGS)
    ap.add_argument("--out", default="data/ami")
    ap.add_argument("--force", action="store_true", help="re-download the zip")
    args = ap.parse_args()

    out = Path(args.out)
    zip_path = out / ZIP_NAME
    if args.force or not zip_path.exists():
        download(zip_path)
    counts = extract(zip_path, out, args.meetings)
    for m in args.meetings:
        print(f"{m}: {counts.get(m, 0)} files" + ("" if m in counts else "  (not found in archive)"))
    print("Data: AMI Meeting Corpus, CC BY 4.0 (https://groups.inf.ed.ac.uk/ami/corpus/)")


if __name__ == "__main__":
    main()
