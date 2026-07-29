"""
logger.py — Structured event logging to SQLite (Stage 4)
=========================================================
Writes all detection / recognition events to a local SQLite database
(events.db) with timestamps. This is the data source for the
"twice-daily summary" feature in the next phase.

Usage:
    from logger import EventLogger
    log = EventLogger()
    log.person_detected(count=2)
    log.face_recognized(name="Shahmir")
    log.unknown_face()

    # Query
    rows = log.today_summary()
"""

import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "events.db")

# ── Schema ────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,          -- ISO-8601
    event_type  TEXT    NOT NULL,          -- person_detected | face_recognized | unknown_face
    person_id   TEXT,                      -- name of recognized person, NULL otherwise
    extra       TEXT                       -- optional free-text metadata
);
"""


class EventLogger:
    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(CREATE_TABLE)
        self._conn.commit()
        print(f"[EventLogger] Database: {db_path}")

    # ── Write helpers ─────────────────────────────────────
    def _write(self, event_type: str, person_id: str = None, extra: str = None):
        ts = datetime.now().isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT INTO events (timestamp, event_type, person_id, extra) "
            "VALUES (?, ?, ?, ?)",
            (ts, event_type, person_id, extra),
        )
        self._conn.commit()

    def person_detected(self, count: int = 1):
        self._write("person_detected", extra=f"count={count}")

    def face_recognized(self, name: str):
        self._write("face_recognized", person_id=name)

    def unknown_face(self):
        self._write("unknown_face")

    # ── Query helpers ─────────────────────────────────────
    def today_summary(self) -> dict:
        """Return a summary dict for today's events."""
        today = date.today().isoformat()
        cur = self._conn.execute(
            "SELECT event_type, COUNT(*) FROM events "
            "WHERE timestamp LIKE ? GROUP BY event_type",
            (f"{today}%",),
        )
        rows = cur.fetchall()
        return {event_type: count for event_type, count in rows}

    def recent_events(self, limit: int = 20) -> list:
        """Return the N most recent event rows as dicts."""
        cur = self._conn.execute(
            "SELECT timestamp, event_type, person_id, extra FROM events "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = ["timestamp", "event_type", "person_id", "extra"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        self._conn.close()


# ── Quick summary CLI ─────────────────────────────────────
if __name__ == "__main__":
    log = EventLogger()
    summary = log.today_summary()
    recent  = log.recent_events(10)

    print("\n=== Today's Summary ===")
    if summary:
        for k, v in summary.items():
            print(f"  {k:25s} : {v}")
    else:
        print("  No events logged today.")

    print("\n=== Last 10 Events ===")
    for row in recent:
        pid = f" ({row['person_id']})" if row["person_id"] else ""
        print(f"  {row['timestamp']}  {row['event_type']}{pid}")
    log.close()
