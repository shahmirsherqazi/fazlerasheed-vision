from logger import EventLogger
log = EventLogger()
print("\n=== Last 20 Events ===")
for row in log.recent_events(20):
    pid = f" ({row['person_id']})" if row["person_id"] else ""
    extra = f"  — {row['extra'][:60]}" if row["extra"] else ""
    print(f"  {row['timestamp']}  {row['event_type']:20s}{pid}{extra}")
log.close()
