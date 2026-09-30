#!/usr/bin/env python3
"""Reset only the local IBVAP admin login to admin / admin123."""
from pathlib import Path
import hashlib
import secrets
import sqlite3


def main():
    db_path = Path(__file__).resolve().parent / "ivap_data" / "ivap.db"
    if not db_path.is_file():
        print("IBVAP database not found. Start IBVAP once, close it, then run this tool again.")
        return 1

    answer = input("This resets only the local admin password to admin123. Type RESET to continue: ")
    if answer.strip() != "RESET":
        print("Cancelled. No changes were made.")
        return 0

    salt = secrets.token_hex(8)
    digest = hashlib.pbkdf2_hmac("sha256", b"admin123", salt.encode(), 100000).hex()
    with sqlite3.connect(str(db_path), timeout=5) as conn:
        table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
        if not table:
            print("The users table is missing. Start IBVAP once, close it, then try again.")
            return 1
        row = conn.execute("SELECT 1 FROM users WHERE u='admin'").fetchone()
        if row:
            conn.execute("UPDATE users SET h=?, role='admin' WHERE u='admin'", (salt + "$" + digest,))
        else:
            conn.execute("INSERT INTO users(u,h,role) VALUES('admin',?,'admin')", (salt + "$" + digest,))
        conn.commit()
    print("Admin login reset. Start IBVAP and sign in as admin / admin123, then change the password.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
