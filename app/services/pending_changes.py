"""Server-side store of prepared-but-not-yet-confirmed SAP changes.

Every chat channel uses the same confirm-before-change flow: a change request
is only *prepared* (the Action Agent's proposal mode - nothing sent to S/4),
saved here under a short one-time code such as CHG-7F3K2Q, and executed only
when that code is confirmed:

  - Joule:  Confirm / Cancel buttons post the text "Confirm change CHG-7F3K2Q"
            (Joule buttons can only send text back)
  - Teams:  the Confirm / Cancel card buttons carry the same code

Because the change itself never travels through the client - only the code -
a confirmation can only ever execute exactly what the server prepared.

Stored on local disk (like the report files), so both uvicorn workers see the
same pending changes. take() claims a record with an atomic rename, so the
same code can never be executed twice - even by two clicks in the same
instant on different workers. Records expire after CONFIRM_TTL_SECONDS.
"""

import json
import os
import re
import secrets
import tempfile
import time

CONFIRM_TTL_SECONDS = 15 * 60
CODE_RE = re.compile(r"\bCHG-[A-Z0-9]{6}\b")
# No 0/O/1/I - easy to read and type back.
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_STORE_DIR = os.path.join(tempfile.gettempdir(), "smart_material_pending")


class PendingChangeError(Exception):
    """The code can't be used; the message is safe to show to the user."""


def _path(code: str) -> str:
    return os.path.join(_STORE_DIR, f"{code}.json")


def _new_code() -> str:
    return "CHG-" + "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))


def save(actions: list[dict], changes: list[str], requested_by: dict | None = None) -> str:
    """Stores prepared actions and returns their one-time confirmation code."""
    os.makedirs(_STORE_DIR, exist_ok=True)
    _sweep_expired()
    record = {
        "actions": actions,
        "changes": changes,
        "requested_by": requested_by or {},
        "created_at": time.time(),
    }
    while True:
        code = _new_code()
        try:
            # "x" = create only if new: never overwrites another pending change.
            with open(_path(code), "x", encoding="utf-8") as handle:
                json.dump(record, handle)
            return code
        except FileExistsError:
            continue


def peek(code: str) -> dict:
    """Reads a pending change without claiming it (e.g. to check who may confirm it).
    Raises PendingChangeError if the code is unknown, already used or expired."""
    code = code.upper()
    if not CODE_RE.fullmatch(code):
        raise PendingChangeError(f"'{code}' is not a valid change code.")
    try:
        with open(_path(code), encoding="utf-8") as handle:
            record = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        raise PendingChangeError(
            f"Change {code} was already confirmed or cancelled, or doesn't exist - nothing more was done."
        ) from None
    if time.time() - record.get("created_at", 0) > CONFIRM_TTL_SECONDS:
        raise PendingChangeError(f"Change {code} has expired - nothing was changed. Please ask for it again.")
    return record


def take(code: str) -> dict:
    """Claims a pending change for execution or cancellation - exactly once.
    Raises PendingChangeError if the code is unknown, already used or expired."""
    code = code.upper()
    if not CODE_RE.fullmatch(code):
        raise PendingChangeError(f"'{code}' is not a valid change code.")
    used = PendingChangeError(
        f"Change {code} was already confirmed or cancelled, or doesn't exist - nothing more was done."
    )
    os.makedirs(_STORE_DIR, exist_ok=True)
    try:
        # O_EXCL create is atomic on Windows and Linux alike: exactly one caller - across
        # threads and worker processes - can create the claim file for a code. (A rename-based
        # claim proved unreliable under concurrency on Windows.) The claim file stays until
        # the expiry sweep, so a code can never be claimed a second time.
        os.close(os.open(f"{_path(code)}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise used from None
    try:
        with open(_path(code), encoding="utf-8") as handle:
            record = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        raise used from None
    _remove(_path(code))
    if time.time() - record.get("created_at", 0) > CONFIRM_TTL_SECONDS:
        raise PendingChangeError(f"Change {code} has expired - nothing was changed. Please ask for it again.")
    return record


def find_code(text: str) -> str | None:
    match = CODE_RE.search((text or "").upper())
    return match.group(0) if match else None


def _sweep_expired() -> None:
    now = time.time()
    for name in os.listdir(_STORE_DIR):
        path = os.path.join(_STORE_DIR, name)
        try:
            if now - os.path.getmtime(path) > CONFIRM_TTL_SECONDS:
                _remove(path)
        except OSError:
            pass


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
