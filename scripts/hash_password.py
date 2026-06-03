"""Generate a bcrypt hash for OMILOG_PASSWORD_HASH.

Usage:
    uv run python scripts/hash_password.py
"""

import getpass
import sys

import bcrypt


def main() -> int:
    pwd = getpass.getpass("New password: ")
    if not pwd:
        print("Empty password, aborting.", file=sys.stderr)
        return 1
    confirm = getpass.getpass("Confirm:      ")
    if pwd != confirm:
        print("Mismatch, aborting.", file=sys.stderr)
        return 1
    if len(pwd.encode("utf-8")) > 72:
        print("bcrypt limit: password must be ≤72 bytes.", file=sys.stderr)
        return 1
    print(bcrypt.hashpw(pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
