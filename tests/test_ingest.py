"""Tests use a tiny hand-made fixture that mimics the AMI NXT layout.

NOTE: this fixture follows the AMI format as documented, but it is NOT real AMI
data. Run `python -m ingest.parse data/ami ES2008a` on the real download and
eyeball the output before trusting the parser.
"""
import textwrap
import zipfile
from pathlib import Path

import pytest

from ingest.download_ami import extract
from ingest.parse import parse_ami, parse_plain
from store.db import add_meeting, connect, get_segments, keyword_search, segment_exists

NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.fixture
def ami_dir(tmp_path):
    d = tmp_path / "ami"
    # Speaker A: two utterances. Note punctuation tokens carry punc="true" and no times.
    write(d / "words/M1.A.words.xml", f"""\
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <nite:root nite:id="M1.A.words" {NS}>
          <w nite:id="M1.A.words0" starttime="1.0" endtime="1.3">Okay</w>
          <w nite:id="M1.A.words1" punc="true">,</w>
          <w nite:id="M1.A.words2" starttime="1.4" endtime="1.7">let's</w>
          <w nite:id="M1.A.words3" starttime="1.7" endtime="2.0">start</w>
          <w nite:id="M1.A.words4" punc="true">.</w>
          <w nite:id="M1.A.words5" starttime="9.0" endtime="9.4">Blue</w>
          <w nite:id="M1.A.words6" starttime="9.4" endtime="9.9">remote</w>
          <w nite:id="M1.A.words7" punc="true">.</w>
        </nite:root>""")
    write(d / "segments/M1.A.segments.xml", f"""\
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <nite:root nite:id="M1.A.segments" {NS}>
          <segment nite:id="M1.A.segments0" transcriber_start="1.0" transcriber_end="2.0">
            <nite:child href="M1.A.words.xml#id(M1.A.words0)..id(M1.A.words4)"/>
          </segment>
          <segment nite:id="M1.A.segments1" transcriber_start="9.0" transcriber_end="9.9">
            <nite:child href="M1.A.words.xml#id(M1.A.words5)..id(M1.A.words7)"/>
          </segment>
        </nite:root>""")
    # Speaker B talks in between; has NO segments file -> exercises the pause fallback.
    write(d / "words/M1.B.words.xml", f"""\
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <nite:root nite:id="M1.B.words" {NS}>
          <w nite:id="M1.B.words0" starttime="4.0" endtime="4.4">Agreed</w>
          <w nite:id="M1.B.words1" punc="true">.</w>
          <w nite:id="M1.B.words2" starttime="6.0" endtime="6.4">Budget</w>
          <w nite:id="M1.B.words3" starttime="6.4" endtime="6.8">twelve</w>
          <w nite:id="M1.B.words4" starttime="20.0" endtime="20.4">Later</w>
        </nite:root>""")
    write(d / "corpusResources/meetings.xml", f"""\
        <?xml version="1.0"?>
        <nite:root nite:id="meetings" {NS}>
          <meeting observation="M1">
            <speaker nxt_agent="A" role="PM"/>
            <speaker nxt_agent="B" role="ME"/>
          </meeting>
        </nite:root>""")
    return d


def test_ami_ordering_text_and_ids(ami_dir):
    segs = parse_ami(ami_dir, "M1")
    assert [s.id for s in segs] == ["M1-s0000", "M1-s0001", "M1-s0002", "M1-s0003", "M1-s0004"]
    assert [s.start for s in segs] == sorted(s.start for s in segs)
    assert segs[0].text == "Okay, let's start."          # punctuation attached, no stray spaces
    assert segs[0].speaker == "A (PM)"                    # role picked up from meetings.xml
    assert segs[1].speaker == "B (ME)" and segs[1].text == "Agreed."   # fallback split at the pause
    assert segs[2].text == "Budget twelve"
    assert segs[3].text == "Blue remote."


def test_ami_missing_meeting_raises_helpful_error(ami_dir):
    with pytest.raises(FileNotFoundError, match="download_ami"):
        parse_ami(ami_dir, "NOPE")


def test_plain_with_and_without_timestamps(tmp_path):
    f = tmp_path / "m.txt"
    f.write_text(
        "[00:00:05] Riya: Let's pick the venue.\n"
        "continued on a second line\n"
        "Aman: Main hall works.\n"
        "[01:10] Riya: Decision: main hall.\n",
        encoding="utf-8",
    )
    segs = parse_plain(f, "club")
    assert [s.speaker for s in segs] == ["Riya", "Aman", "Riya"]
    assert segs[0].text == "Let's pick the venue. continued on a second line"
    assert segs[0].start == 5.0 and segs[2].start == 70.0
    assert segs[1].start > segs[0].start                  # invented timestamps stay monotonic
    assert segs[1].end <= segs[2].start


def test_db_roundtrip_fts_and_replace(ami_dir):
    conn = connect(":memory:")
    segs = parse_ami(ami_dir, "M1")
    assert add_meeting(conn, "M1", "Test", "AMI Meeting Corpus", "CC BY 4.0", segs) == 5
    assert [r["id"] for r in get_segments(conn, "M1")] == [s.id for s in segs]
    assert segment_exists(conn, "M1-s0003") and not segment_exists(conn, "M1-s9999")

    hits = keyword_search(conn, "what was the blue remote budget?")
    assert hits and {"M1-s0003", "M1-s0002"} <= {h["id"] for h in hits}
    assert keyword_search(conn, "???") == []

    # Re-adding replaces cleanly (cascade delete, FTS stays consistent).
    add_meeting(conn, "M1", "Test", "AMI Meeting Corpus", "CC BY 4.0", segs[:2])
    assert len(get_segments(conn, "M1")) == 2
    assert keyword_search(conn, "budget") == []


def test_item_must_cite_existing_segment(ami_dir):
    import sqlite3
    conn = connect(":memory:")
    add_meeting(conn, "M1", "Test", "x", None, parse_ami(ami_dir, "M1"))
    conn.execute("INSERT INTO items(meeting_id,type,text,cited_segment_id) VALUES ('M1','decision','Go blue','M1-s0003')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO items(meeting_id,type,text,cited_segment_id) VALUES ('M1','decision','Ghost','M1-s9999')")


def test_download_extract_filters_files(tmp_path):
    zp = tmp_path / "ann.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("words/ES2008a.A.words.xml", "<x/>")
        z.writestr("segments/ES2008a.A.segments.xml", "<x/>")
        z.writestr("words/ES2009a.A.words.xml", "<x/>")           # not requested
        z.writestr("corpusResources/meetings.xml", "<x/>")
    out = tmp_path / "out"
    assert extract(zp, out, ["ES2008a"]) == {"ES2008a": 2}
    assert (out / "words/ES2008a.A.words.xml").exists()
    assert (out / "corpusResources/meetings.xml").exists()
    assert not (out / "words/ES2009a.A.words.xml").exists()
