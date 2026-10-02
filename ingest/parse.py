"""Turn transcripts into timestamped, speaker-labelled segments.

A segment is the unit everything else cites: extracted items and answers point
at ids like "ES2008a-s0042", and the verify step drops anything that cites an
id that does not exist.

    python -m ingest.parse data/ami ES2008a      # eyeball real AMI output
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

GAP_SECONDS = 1.0        # pause that starts a new utterance when no segment file covers the words
MAX_WORDS = 60
WORDS_PER_SECOND = 2.5   # only used to estimate timing for untimed plain-text lines


@dataclass(frozen=True)
class Segment:
    id: str
    meeting_id: str
    idx: int
    speaker: str
    start: float
    end: float
    text: str


def segment_id(meeting_id: str, idx: int) -> str:
    return f"{meeting_id}-s{idx:04d}"


# ------------------------------------------------------------------ AMI

@dataclass
class _Tok:
    id: str | None
    start: float | None
    end: float | None
    text: str
    punc: bool

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _nite_id(el: ET.Element) -> str | None:
    for k, v in el.attrib.items():
        if k == "id" or k.endswith("}id"):
            return v
    return None


def _float(v: str | None) -> float | None:
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def _read_tokens(path: Path) -> list[_Tok]:
    """Words in file order. Punctuation tokens (punc="true") carry no times and
    are glued to the previous word when the text is built."""
    toks = []
    for el in ET.parse(path).getroot().iter():
        if _local(el.tag) != "w":
            continue
        text = (el.text or "").strip()
        if text:
            toks.append(_Tok(_nite_id(el), _float(el.get("starttime")), _float(el.get("endtime")),
                             text, el.get("punc") == "true"))
    return toks


def _join(toks: list[_Tok]) -> str:
    out: list[str] = []
    for t in toks:
        if t.punc and out:
            out[-1] += t.text
        else:
            out.append(t.text)
    return " ".join(out)


def _utterance(toks: list[_Tok]) -> tuple[float, float, str] | None:
    timed = [t for t in toks if t.timed]
    if not timed:
        return None
    return min(t.start for t in timed), max(t.end for t in timed), _join(toks)


def _split_by_pause(toks: list[_Tok]) -> list[list[_Tok]]:
    groups: list[list[_Tok]] = []
    cur: list[_Tok] = []
    last_end: float | None = None
    n = 0
    for t in toks:
        if t.timed and cur and ((last_end is not None and t.start - last_end > GAP_SECONDS) or n >= MAX_WORDS):
            groups.append(cur)
            cur, n = [], 0
        cur.append(t)
        if t.timed:
            last_end = t.end
        if not t.punc:
            n += 1
    if cur:
        groups.append(cur)
    return groups


_HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def _read_segment_ranges(path: Path):
    """Yield ([(first_word_id, last_word_id), ...], transcriber_start, transcriber_end)."""
    for seg in ET.parse(path).getroot().iter():
        if _local(seg.tag) != "segment":
            continue
        ranges = []
        for child in seg.iter():
            if _local(child.tag) == "child":
                for m in _HREF.finditer(child.get("href", "")):
                    ranges.append((m.group(1), m.group(2) or m.group(1)))
        yield ranges, _float(seg.get("transcriber_start")), _float(seg.get("transcriber_end"))


def _speaker_roles(meetings_xml: Path, meeting_id: str) -> dict[str, str]:
    if not meetings_xml.exists():
        return {}
    for m in ET.parse(meetings_xml).getroot().iter():
        if _local(m.tag) == "meeting" and m.get("observation") == meeting_id:
            return {s.get("nxt_agent"): s.get("role")
                    for s in m.iter()
                    if _local(s.tag) == "speaker" and s.get("nxt_agent") and s.get("role")}
    return {}


def parse_ami(ami_dir: str | Path, meeting_id: str) -> list[Segment]:
    """AMI NXT annotations -> segments.

    Reads <ami_dir>/words/<meeting>.<speaker>.words.xml. If a matching
    segments/<meeting>.<speaker>.segments.xml exists, the transcribers' own
    utterance boundaries are used; any words it does not cover (or all words,
    when there is no segments file) are split at pauses. Speaker labels include
    the role from corpusResources/meetings.xml, e.g. "A (PM)".
    """
    ami_dir = Path(ami_dir)
    files = sorted((ami_dir / "words").glob(f"{meeting_id}.*.words.xml"))
    if not files:
        raise FileNotFoundError(
            f"No word files for {meeting_id} under {ami_dir / 'words'}. "
            f"Run: python -m ingest.download_ami --meetings {meeting_id}")
    roles = _speaker_roles(ami_dir / "corpusResources" / "meetings.xml", meeting_id)

    raw: list[tuple[float, float, str, str]] = []  # start, end, speaker, text
    for f in files:
        agent = f.name.split(".")[1]
        speaker = f"{agent} ({roles[agent]})" if agent in roles else agent
        toks = _read_tokens(f)
        covered: set[int] = set()

        seg_file = ami_dir / "segments" / f"{meeting_id}.{agent}.segments.xml"
        if seg_file.exists():
            pos = {t.id: i for i, t in enumerate(toks) if t.id}
            for ranges, ts, te in _read_segment_ranges(seg_file):
                idxs: set[int] = set()
                for a, b in ranges:
                    if a in pos and b in pos:
                        lo, hi = sorted((pos[a], pos[b]))
                        idxs.update(range(lo, hi + 1))
                if not idxs:
                    continue
                covered |= idxs
                utt = _utterance([toks[i] for i in sorted(idxs)])
                if utt:
                    raw.append((utt[0], utt[1], speaker, utt[2]))
                elif ts is not None and te is not None:
                    raw.append((ts, te, speaker, _join([toks[i] for i in sorted(idxs)])))

        rest = [t for i, t in enumerate(toks) if i not in covered]
        for group in _split_by_pause(rest):
            utt = _utterance(group)
            if utt:
                raw.append((utt[0], utt[1], speaker, utt[2]))

    raw.sort(key=lambda r: (r[0], r[2]))
    return [Segment(segment_id(meeting_id, i), meeting_id, i, spk, s, e, text)
            for i, (s, e, spk, text) in enumerate(raw)]


# ------------------------------------------------------------------ plain text

_LINE = re.compile(
    r"^(?:\[(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)\]\s*)?(?P<spk>[^:\[\]]{1,40}):\s+(?P<text>.+)$")


def _estimate(text: str) -> float:
    return max(1.0, len(text.split()) / WORDS_PER_SECOND)


def parse_plain(path: str | Path, meeting_id: str) -> list[Segment]:
    """One utterance per line:  [hh:mm:ss] Speaker: text   or   [mm:ss] Speaker: text.
    The timestamp is optional; untimed lines get an estimated start after the
    previous utterance. Lines that do not look like "Speaker: text" continue
    the previous utterance."""
    rows: list[list] = []  # [timestamp or None, speaker, text]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if m:
            ts = None
            if m["m"] is not None:
                ts = float(int(m["h"] or 0) * 3600 + int(m["m"]) * 60 + int(m["s"]))
            rows.append([ts, m["spk"].strip(), m["text"].strip()])
        elif rows:
            rows[-1][2] += " " + line
    if not rows:
        raise ValueError(f"No utterances found in {path}")

    starts: list[float] = []
    for ts, _, text in rows:
        if ts is not None:
            start = ts
        elif starts:
            start = starts[-1] + _estimate(rows[len(starts) - 1][2])
        else:
            start = 0.0
        starts.append(max(start, starts[-1]) if starts else start)

    out = []
    for i, ((_, spk, text), start) in enumerate(zip(rows, starts)):
        nxt = starts[i + 1] if i + 1 < len(starts) else start + _estimate(text)
        out.append(Segment(segment_id(meeting_id, i), meeting_id, i, spk, start,
                           max(start, min(start + _estimate(text), nxt)), text))
    return out


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: python -m ingest.parse <ami_dir> <meeting_id> [max_lines]")
    segs = parse_ami(sys.argv[1], sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    for s in segs[:limit]:
        print(f"{s.id}  [{int(s.start // 60):02d}:{int(s.start % 60):02d}] {s.speaker}: {s.text}")
    print(f"... {len(segs)} segments, {segs[-1].end / 60:.1f} min, "
          f"speakers: {', '.join(sorted({s.speaker for s in segs}))}")


if __name__ == "__main__":
    main()
