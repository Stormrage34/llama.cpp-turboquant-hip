#!/usr/bin/env python3
"""Simple telemetry parser for rocprofv3 SQLite output.
Usage: python3 scripts/parse_telemetry.py <counters.sqlite>
Prints all counter name/value pairs.
"""
import sys, os, sqlite3

def main():
    if len(sys.argv) < 2:
        print("Usage: parse_telemetry.py <counters.sqlite>")
        sys.exit(1)
    db_path = sys.argv[1]
    if not os.path.isfile(db_path):
        print(f"Error: file not found: {db_path}")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT name, value FROM counters ORDER BY name")
        rows = cur.fetchall()
        if not rows:
            print("No counters found in the database.")
            return
        print("=== Counter values ===")
        for name, value in rows:
            print(f"{name}: {value}")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
