PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meetings (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    source      TEXT,            -- e.g. 'AMI', 'MeetingBank', 'synthetic'
    license     TEXT,            -- keep attribution with the data
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS segments (
    id          TEXT PRIMARY KEY,                 -- e.g. 'ES2008a-s0042'
    meeting_id  TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    speaker     TEXT NOT NULL,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    text        TEXT NOT NULL,
    UNIQUE (meeting_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_segments_meeting ON segments(meeting_id, idx);

-- Extracted decisions / action items / open questions.
-- cited_segment_id is NOT NULL and a foreign key: an item that cites a
-- segment that does not exist cannot be stored.
CREATE TABLE IF NOT EXISTS items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id        TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    type              TEXT NOT NULL CHECK (type IN ('decision', 'action', 'question')),
    text              TEXT NOT NULL,
    owner             TEXT,                       -- a real speaker or 'Unassigned'
    deadline          TEXT,                       -- only if spoken in the meeting
    cited_segment_id  TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    status            TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_items_meeting ON items(meeting_id, type);

-- Keyword (BM25) search over segment text.
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text, speaker UNINDEXED, content='segments', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text, speaker) VALUES (new.rowid, new.text, new.speaker);
END;
CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text, speaker)
    VALUES ('delete', old.rowid, old.text, old.speaker);
END;
