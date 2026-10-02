"""Parse a transcript and store it.

    python -m ingest.add_meeting ami  data/ami ES2008a
    python -m ingest.add_meeting plain my_meeting.txt --id hinglish-01 --title "Club planning (synthetic)"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ingest.parse import parse_ami, parse_plain
from store.db import add_meeting, connect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    a = sub.add_parser("ami")
    a.add_argument("ami_dir")
    a.add_argument("meeting_id")
    p = sub.add_parser("plain")
    p.add_argument("path")
    p.add_argument("--id")
    for sp in (a, p):
        sp.add_argument("--title")
        sp.add_argument("--db", default="data/meetings.db")
    args = ap.parse_args()

    if args.kind == "ami":
        mid = args.meeting_id
        segs = parse_ami(args.ami_dir, mid)
        source, lic = "AMI Meeting Corpus", "CC BY 4.0"
    else:
        mid = args.id or Path(args.path).stem
        segs = parse_plain(args.path, mid)
        source, lic = "self-written / synthetic", None

    conn = connect(args.db)
    n = add_meeting(conn, mid, args.title or mid, source, lic, segs)
    speakers = sorted({s.speaker for s in segs})
    print(f"stored {mid}: {n} segments, {len(speakers)} speakers ({', '.join(speakers)})")


if __name__ == "__main__":
    main()
