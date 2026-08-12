"""
logger.py — Structured event logging to SQLite (Stages 4 & 5)
==============================================================
Writes all detection / tracking / activity / LLM events to a local SQLite
database (events.db) with timestamps.

Event types
-----------
  person_detected   — YOLO+ByteTrack saw N persons in frame
  face_recognized   — insightface matched a known person (name in person_id)
  unknown_face      — insightface saw a face but couldn't match it
  activity_detected — LLM returned an activity label for a tracked person
                      (person_id = ByteTrack ID as string, extra = label)
  high_priority     — activity label contained a high-priority keyword
                      (drawer / fridge / etc.)
  llm_narration     — generic LLM text (scene desc or shift summary)

Usage:
    from logger import EventLogger
    log = EventLogger()
    log.person_detected(count=2)
    log.activity_detected(label="opening a drawer", person_id="3")
    log.high_priority_alert("ID=3 may be 'drawer': opening a drawer")
"""

import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "events.db")

# ── Schema ────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    person_id   TEXT,
    extra       TEXT
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
            "INSERT INTO events (timestamp, event_type, person_id, extra) VALUES (?, ?, ?, ?)",
            (ts, event_type, person_id, extra),
        )
        self._conn.commit()

    def person_detected(self, count: int = 1):
        """Log that N persons were visible in a frame."""
        self._write("person_detected", extra=f"count={count}")

    def face_recognized(self, name: str):
        """Log a successful face match (Stage 3)."""
        self._write("face_recognized", person_id=name)

    def unknown_face(self):
        """Log an unrecognised face (Stage 3)."""
        self._write("unknown_face")

    def activity_detected(self, label: str, person_id: str = None):
        """
        Log an activity label returned by the vision LLM for one person.

        Parameters
        ----------
        label     : str   — e.g. "opening a drawer", "sitting"
        person_id : str   — ByteTrack ID (as string) for the tracked person
        """
        self._write("activity_detected", person_id=person_id, extra=label)

    def high_priority_alert(self, detail: str):
        """
        Log a high-priority event (e.g. drawer/fridge keyword detected).

        Parameters
        ----------
        detail : str — full alert message
        """
        self._write("high_priority", extra=detail)

    def llm_narration(self, description: str):
        """Store a generic scene description or shift summary from the LLM."""
        self._write("llm_narration", extra=description)

    # ── Query helpers ─────────────────────────────────────
    def today_summary(self) -> dict:
        """Return a count-per-event-type dict for today."""
        today = date.today().isoformat()
        cur = self._conn.execute(
            "SELECT event_type, COUNT(*) FROM events WHERE timestamp LIKE ? GROUP BY event_type",
            (f"{today}%",),
        )
        return {et: cnt for et, cnt in cur.fetchall()}

    def recent_events(self, limit: int = 20) -> list:
        """Return the N most recent event rows as dicts."""
        cur = self._conn.execute(
            "SELECT timestamp, event_type, person_id, extra FROM events "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = ["timestamp", "event_type", "person_id", "extra"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def today_narrations(self) -> list:
        """Return all activity_detected rows for today (used for shift summary)."""
        today = date.today().isoformat()
        cur = self._conn.execute(
            "SELECT timestamp, person_id, extra FROM events "
            "WHERE event_type='activity_detected' AND timestamp LIKE ? "
            "ORDER BY id ASC",
            (f"{today}%",),
        )
        return [{"timestamp": r[0], "person_id": r[1], "extra": r[2]}
                for r in cur.fetchall()]

    def today_high_priority(self) -> list:
        """Return all high-priority alerts for today."""
        today = date.today().isoformat()
        cur = self._conn.execute(
            "SELECT timestamp, extra FROM events "
            "WHERE event_type='high_priority' AND timestamp LIKE ? "
            "ORDER BY id ASC",
            (f"{today}%",),
        )
        return [{"timestamp": r[0], "detail": r[1]} for r in cur.fetchall()]

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
        print(f"  {row['timestamp']}  {row['event_type']}{pid}"
              + (f"  — {row['extra'][:60]}" if row["extra"] else ""))

    alerts = log.today_high_priority()
    if alerts:
        print(f"\n=== ⚠  High-Priority Alerts Today ({len(alerts)}) ===")
        for a in alerts:
            print(f"  {a['timestamp']}  {a['detail']}")

    log.close()
