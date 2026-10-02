from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

from ingest.parse import Segment

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_STOPWORDS = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "it",
              "we", "did", "do", "what", "about", "that", "this", "was", "were", "who"}


DEFAULT_DB = "data/meetings.db"


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    """Open the database (creating the schema if needed). Use ':memory:' in tests."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def add_meeting(conn, meeting_id: str, title: str | None, source: str | None,
                license: str | None, segments: Iterable[Segment]) -> int:
    """Store a meeting and its segments; returns the segment count.
    Re-adding a meeting replaces its segments (and, via cascade, its items)."""
    conn.execute(
        """INSERT INTO meetings (id, title, source, license) VALUES (?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET title=excluded.title,
               source=excluded.source, license=excluded.license""",
        (meeting_id, title, source, license))
    conn.execute("DELETE FROM segments WHERE meeting_id = ?", (meeting_id,))
    rows = [(s.id, s.meeting_id, s.idx, s.speaker, s.start, s.end, s.text) for s in segments]
    conn.executemany(
        "INSERT INTO segments (id, meeting_id, idx, speaker, start_s, end_s, text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


def get_segments(conn, meeting_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM segments WHERE meeting_id = ? ORDER BY idx", (meeting_id,)).fetchall()


def get_segment(conn, segment_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()


def segment_exists(conn, segment_id: str) -> bool:
    return get_segment(conn, segment_id) is not None


def neighbours(conn, segment_id: str, before: int = 2, after: int = 2) -> list[sqlite3.Row]:
    """The segment plus the turns around it, for context windows."""
    seg = get_segment(conn, segment_id)
    if seg is None:
        return []
    return conn.execute(
        "SELECT * FROM segments WHERE meeting_id = ? AND idx BETWEEN ? AND ? ORDER BY idx",
        (seg["meeting_id"], seg["idx"] - before, seg["idx"] + after)).fetchall()


def keyword_search(conn, query: str, k: int = 10) -> list[sqlite3.Row]:
    """BM25 over segment text. Lower score = better match."""
    terms = [t for t in re.findall(r"\w+", query.lower()) if t not in _STOPWORDS]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    return conn.execute(
        """SELECT s.*, bm25(segments_fts) AS score
           FROM segments_fts JOIN segments s ON s.rowid = segments_fts.rowid
           WHERE segments_fts MATCH ? ORDER BY score LIMIT ?""", (match, k)).fetchall()


def add_item(conn, meeting_id: str, type_: str, text: str, cited_segment_id: str,
             owner: str | None = None, deadline: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO items (meeting_id, type, text, owner, deadline, cited_segment_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (meeting_id, type_, text, owner, deadline, cited_segment_id))
    conn.commit()
    return int(cur.lastrowid)


def list_items(conn, type_: str | None = None, meeting_id: str | None = None,
               owner: str | None = None, status: str | None = "open") -> list[sqlite3.Row]:
    sql = ("SELECT i.*, s.speaker AS cited_speaker, s.start_s AS cited_start, s.text AS cited_text "
           "FROM items i JOIN segments s ON s.id = i.cited_segment_id WHERE 1=1")
    args: list = []
    for col, val in (("i.type", type_), ("i.meeting_id", meeting_id),
                     ("i.owner", owner), ("i.status", status)):
        if val is not None:
            sql += f" AND {col} = ?"
            args.append(val)
    return conn.execute(sql + " ORDER BY i.meeting_id, s.idx", args).fetchall()


def clear_items(conn, meeting_id: str) -> None:
    """Remove a meeting's extracted items so extraction can be re-run cleanly."""
    conn.execute("DELETE FROM items WHERE meeting_id = ?", (meeting_id,))
    conn.commit()
