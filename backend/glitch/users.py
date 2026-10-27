"""Accounts, sessions, plans and daily quotas — SQLite, standard library only.

Implements BUSINESS_MODEL.md §6 (plans and metering) and §8.3 (subscription
states) without a payment provider: an admin grants or changes plans, and the
state machine (active → 48 h grace → expired; cancelled → expired at period
end) is applied lazily whenever an account is read.

Storage is ``instance/glitch.db`` (owner-only permissions). Schema changes go
through ``MIGRATIONS``; ``PRAGMA user_version`` records how many have run.

Security notes:
* passwords: scrypt (n=2^14, r=8, p=1) with a per-user salt;
* sessions: random 32-byte tokens; only their SHA-256 is stored;
* quotas: reserved atomically inside ``BEGIN IMMEDIATE`` before the model call
  and released if the call fails (§6.3 rules 4 and 9).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config
from .errors import DetectorError
from .prompt import TONES

# -- plans (BUSINESS_MODEL.md §6.1) ------------------------------------------
FREE_TONES = ("natural", "academic")
PLANS = {
    "free": {
        "label": "Free", "max_words": 500, "rewrite_words": 500, "rewrites_per_day": 5,
        "checks_per_day": 100, "tones": FREE_TONES, "ads": True, "seats": 1,
    },
    "premium": {
        "label": "Premium", "max_words": 2500, "rewrite_words": config.MAX_REWRITE_WORDS,
        "rewrites_per_day": 15,
        "checks_per_day": 500, "tones": tuple(TONES), "ads": False, "seats": 1,
    },
    "family": {
        "label": "Family", "max_words": 2500, "rewrite_words": config.MAX_REWRITE_WORDS,
        "rewrites_per_day": 15,
        "checks_per_day": 500, "tones": tuple(TONES), "ads": False, "seats": 5,
    },
}
# Visitors without an account can check text but not rewrite it.
ANONYMOUS = {**PLANS["free"], "label": "Guest", "rewrites_per_day": 0}

STATUSES = ("free", "pending_payment", "active", "grace", "expired", "cancelled")
PAID_ACCESS = ("active", "grace", "cancelled")  # cancelled keeps access until period end
GRACE_SECONDS = 48 * 3600
CAT = timezone(timedelta(hours=2))  # quotas reset at 00:00 CAT (§6.3 rule 3)

SESSION_COOKIE = "glitch_session"
SESSION_DAYS = 30
MAX_FAILED_LOGINS = 5
FAILED_LOGIN_WINDOW_S = 15 * 60

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
_QUOTA_COLUMNS = {"checks", "rewrites"}

MIGRATIONS = [
    # 1 — accounts, sessions, daily usage
    """
    CREATE TABLE users (
        id            TEXT PRIMARY KEY,
        email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
        name          TEXT NOT NULL DEFAULT '',
        password_hash TEXT NOT NULL,
        plan          TEXT NOT NULL DEFAULT 'free',
        status        TEXT NOT NULL DEFAULT 'free',
        period_end    INTEGER,
        suspended     INTEGER NOT NULL DEFAULT 0,
        notes         TEXT NOT NULL DEFAULT '',
        created_at    INTEGER NOT NULL,
        last_seen     INTEGER
    );
    CREATE TABLE sessions (
        token_hash TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at INTEGER NOT NULL,
        expires_at INTEGER NOT NULL
    );
    CREATE INDEX sessions_user ON sessions(user_id);
    CREATE TABLE usage (
        subject  TEXT NOT NULL,
        day      TEXT NOT NULL,
        checks   INTEGER NOT NULL DEFAULT 0,
        rewrites INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (subject, day)
    );
    """,
]


def cat_today() -> str:
    return datetime.now(CAT).date().isoformat()


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$16384$8$1$%s$%s" % (
        base64.b64encode(salt).decode(), base64.b64encode(digest).decode()
    )


def _check_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


_DUMMY_HASH = _hash_password(secrets.token_hex(8))  # equalises timing for unknown emails


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _clean_email(email: str) -> str:
    email = (email or "").strip().lower()
    if len(email) > 254 or not _EMAIL_RE.match(email):
        raise DetectorError("Enter a valid email address.", status=400, code="invalid_email")
    return email


def _check_new_password(password: str) -> str:
    password = password or ""
    if len(password) < 8:
        raise DetectorError("Use a password of at least 8 characters.", status=400, code="weak_password")
    if len(password) > 128:
        raise DetectorError("That password is too long (128 characters at most).", status=400)
    return password


class UserStore:
    def __init__(self, path: Path | None = None):
        self._path = path or (config.INSTANCE_DIR / "glitch.db")
        self._migrated = False
        self._lock = threading.Lock()
        self._failed: dict[str, list[float]] = {}

    # -- database ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if not self._migrated:
            with self._lock:
                if not self._migrated:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    if not self._path.exists():
                        os.close(os.open(self._path, os.O_WRONLY | os.O_CREAT, 0o600))
                    conn = sqlite3.connect(self._path, timeout=10)
                    try:
                        self.migrate(conn)
                    finally:
                        conn.close()
                    self._migrated = True
        conn = sqlite3.connect(self._path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def migrate(conn: sqlite3.Connection) -> int:
        """Apply pending migrations; returns the schema version."""
        conn.execute("PRAGMA journal_mode = WAL")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
            conn.executescript(f"BEGIN; {script} PRAGMA user_version = {number}; COMMIT;")
        return len(MIGRATIONS)

    @contextmanager
    def _tx(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _row(self, sql: str, *args):
        conn = self._connect()
        try:
            return conn.execute(sql, args).fetchone()
        finally:
            conn.close()

    # -- plans and states ---------------------------------------------------
    def _settle(self, row: sqlite3.Row) -> dict:
        """Apply time-based state changes (§8.3) and return a plain dict."""
        user = dict(row)
        now = time.time()
        end = user["period_end"]
        status = user["status"]
        if user["plan"] != "free" and end:
            if status == "active" and now > end:
                status = "grace" if now <= end + GRACE_SECONDS else "expired"
            elif status == "grace" and now > end + GRACE_SECONDS:
                status = "expired"
            elif status == "cancelled" and now > end:
                status = "expired"
        if status != user["status"]:
            with self._tx() as conn:
                conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user["id"]))
            user["status"] = status
        return user

    @staticmethod
    def effective_plan(user: dict | None) -> str:
        if user is None:
            return "guest"
        if user["plan"] in PLANS and user["status"] in PAID_ACCESS:
            return user["plan"]
        return "free"

    @classmethod
    def entitlements(cls, user: dict | None) -> dict:
        plan_id = cls.effective_plan(user)
        plan = ANONYMOUS if plan_id == "guest" else PLANS[plan_id]
        return {"id": plan_id, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in plan.items()}}

    @classmethod
    def public(cls, user: dict) -> dict:
        """Safe view of an account. Never includes the password hash."""
        return {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "plan": user["plan"],
            "status": user["status"],
            "effective_plan": cls.effective_plan(user),
            "period_end": user["period_end"],
            "suspended": bool(user["suspended"]),
            "notes": user.get("notes", ""),
            "created_at": user["created_at"],
            "last_seen": user["last_seen"],
        }

    # -- accounts -----------------------------------------------------------
    def create(self, email: str, password: str, name: str = "") -> dict:
        email = _clean_email(email)
        password = _check_new_password(password)
        user_id = secrets.token_hex(8)
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO users (id, email, name, password_hash, created_at, last_seen)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, email, (name or "").strip()[:60], _hash_password(password),
                     int(time.time()), int(time.time())),
                )
        except sqlite3.IntegrityError:
            raise DetectorError(
                "An account with that email already exists. Sign in instead.",
                status=409, code="email_taken",
            ) from None
        return self.get(user_id)

    def get(self, user_id: str) -> dict | None:
        row = self._row("SELECT * FROM users WHERE id = ?", user_id)
        return self._settle(row) if row else None

    def authenticate(self, email: str, password: str) -> dict:
        email = (email or "").strip().lower()
        now = time.time()
        with self._lock:
            recent = [t for t in self._failed.get(email, []) if now - t < FAILED_LOGIN_WINDOW_S]
            self._failed[email] = recent
        if len(recent) >= MAX_FAILED_LOGINS:
            raise DetectorError(
                "Too many attempts. Wait a few minutes and try again.", status=429, code="locked"
            )
        row = self._row("SELECT * FROM users WHERE email = ?", email)
        ok = _check_password(password or "", row["password_hash"] if row else _DUMMY_HASH)
        if not row or not ok:
            with self._lock:
                self._failed.setdefault(email, []).append(now)
            raise DetectorError("That email and password don't match.", status=401, code="bad_login")
        if row["suspended"]:
            raise DetectorError(
                "This account is suspended. Contact support.", status=403, code="suspended"
            )
        with self._lock:
            self._failed.pop(email, None)
        return self._settle(row)

    def change_password(self, user_id: str, current: str, new: str) -> None:
        row = self._row("SELECT password_hash FROM users WHERE id = ?", user_id)
        if not row or not _check_password(current or "", row["password_hash"]):
            raise DetectorError("Your current password isn't right.", status=401, code="bad_login")
        new = _check_new_password(new)
        with self._tx() as conn:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(new), user_id))

    def update(self, user_id: str, *, clear_period: bool = False, **fields) -> dict:
        """Admin/profile update of name, plan, status, period_end, suspended, notes.

        ``None`` values are ignored; pass ``clear_period=True`` to remove the
        period end (an open-ended plan).
        """
        allowed = {"name", "plan", "status", "period_end", "suspended", "notes"}
        changes = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if clear_period:
            changes["period_end"] = None
        if "plan" in changes and changes["plan"] not in PLANS:
            raise DetectorError("Unknown plan.", status=400)
        if "status" in changes and changes["status"] not in STATUSES:
            raise DetectorError("Unknown subscription status.", status=400)
        if "name" in changes:
            changes["name"] = str(changes["name"]).strip()[:60]
        if "notes" in changes:
            changes["notes"] = str(changes["notes"]).strip()[:500]
        if "suspended" in changes:
            changes["suspended"] = 1 if changes["suspended"] else 0
        if changes.get("plan") == "free":
            changes.setdefault("status", "free")
            changes["period_end"] = None
        if changes:
            assignments = ", ".join(f"{column} = ?" for column in changes)
            with self._tx() as conn:
                cur = conn.execute(
                    f"UPDATE users SET {assignments} WHERE id = ?", (*changes.values(), user_id)
                )
                if cur.rowcount == 0:
                    raise DetectorError("No such user.", status=404)
                if changes.get("suspended"):
                    conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        user = self.get(user_id)
        if user is None:
            raise DetectorError("No such user.", status=404)
        return user

    def grant(self, user_id: str, plan: str, days: int) -> dict:
        """Start or extend a paid plan by ``days`` from the later of now or the current end."""
        if plan not in PLANS or plan == "free":
            raise DetectorError("Choose Premium or Family.", status=400)
        if not 1 <= int(days) <= 3660:
            raise DetectorError("Grant between 1 and 3,660 days.", status=400)
        user = self.get(user_id)
        if user is None:
            raise DetectorError("No such user.", status=404)
        now = int(time.time())
        current = user["period_end"] if user["plan"] == plan and user["status"] in PAID_ACCESS else None
        start = max(now, current or now)
        return self.update(user_id, plan=plan, status="active", period_end=start + int(days) * 86400)

    def delete(self, user_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.execute("DELETE FROM usage WHERE subject = ?", (user_id,))
        return cur.rowcount > 0

    def list(self, *, query: str = "", plan: str = "", status: str = "",
             limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        where, args = [], []
        if query:
            where.append("(email LIKE ? OR name LIKE ?)")
            args += [f"%{query.strip()}%"] * 2
        if plan in PLANS:
            where.append("plan = ?")
            args.append(plan)
        if status in STATUSES:
            where.append("status = ?")
            args.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        conn = self._connect()
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM users {clause}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM users {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*args, max(1, min(int(limit), 200)), max(0, int(offset))),
            ).fetchall()
        finally:
            conn.close()
        return [self._settle(row) for row in rows], total

    def summary(self) -> dict:
        """Account counts for the admin analytics view."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM users").fetchall()
            active_today = conn.execute(
                "SELECT COUNT(*) FROM usage u JOIN users s ON s.id = u.subject WHERE u.day = ?",
                (cat_today(),),
            ).fetchone()[0]
        finally:
            conn.close()
        users = [self._settle(row) for row in rows]
        month_ago = time.time() - 30 * 86400
        by_plan: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for user in users:
            plan = self.effective_plan(user)
            by_plan[plan] = by_plan.get(plan, 0) + 1
            by_status[user["status"]] = by_status.get(user["status"], 0) + 1
        paid = sum(n for plan, n in by_plan.items() if plan != "free")
        return {
            "connected": True,
            "total": len(users),
            "paid": paid,
            "free": by_plan.get("free", 0),
            "by_plan": by_plan,
            "by_status": by_status,
            "new_month": sum(1 for u in users if u["created_at"] >= month_ago),
            "active_today": active_today,
            "suspended": sum(1 for u in users if u["suspended"]),
            "payments": False,
            "note": "Payments aren't connected yet; plans are granted from the Users view.",
        }

    # -- sessions -----------------------------------------------------------
    def start_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self._tx() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (_token_hash(token), user_id, now, now + SESSION_DAYS * 86400),
            )
        return token

    def user_for_session(self, token: str) -> dict | None:
        if not token or len(token) > 128:
            return None
        row = self._row(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id"
            " WHERE s.token_hash = ? AND s.expires_at > ?",
            _token_hash(token), int(time.time()),
        )
        if not row or row["suspended"]:
            return None
        now = int(time.time())
        if (row["last_seen"] or 0) < now - 300:  # don't write on every request
            with self._tx() as conn:
                conn.execute("UPDATE users SET last_seen = ? WHERE id = ?", (now, row["id"]))
        return self._settle(row)

    def end_session(self, token: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token or ""),))

    def end_all_sessions(self, user_id: str) -> int:
        with self._tx() as conn:
            return conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,)).rowcount

    # -- quotas -------------------------------------------------------------
    @contextmanager
    def reserve(self, subject: str, kind: str, limit: int):
        """Count one ``kind`` use for today, refusing past ``limit``.

        The count is released if the body raises, so failed requests don't
        use up quota (§6.3 rule 4). Reservation is atomic (rule 9).
        """
        if kind not in _QUOTA_COLUMNS:
            raise ValueError(kind)
        day = cat_today()
        with self._tx() as conn:
            conn.execute("INSERT OR IGNORE INTO usage (subject, day) VALUES (?, ?)", (subject, day))
            cur = conn.execute(
                f"UPDATE usage SET {kind} = {kind} + 1 WHERE subject = ? AND day = ? AND {kind} < ?",
                (subject, day, limit),
            )
            if cur.rowcount == 0:
                noun = "rewrites" if kind == "rewrites" else "checks"
                raise DetectorError(
                    f"You've used today's {limit} {noun}. They reset at midnight (CAT).",
                    status=429, code="quota",
                )
        try:
            yield
        except BaseException:
            with self._tx() as conn:
                conn.execute(
                    f"UPDATE usage SET {kind} = MAX({kind} - 1, 0) WHERE subject = ? AND day = ?",
                    (subject, day),
                )
            raise

    def usage_today(self, subject: str) -> dict:
        row = self._row("SELECT checks, rewrites FROM usage WHERE subject = ? AND day = ?",
                        subject, cat_today())
        return {"checks": row["checks"], "rewrites": row["rewrites"]} if row else {"checks": 0, "rewrites": 0}

    def usage_history(self, subject: str, days: int = 30) -> list[dict]:
        start = (datetime.now(CAT).date() - timedelta(days=days - 1)).isoformat()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT day, checks, rewrites FROM usage WHERE subject = ? AND day >= ? ORDER BY day",
                (subject, start),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def reset_today(self, subject: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM usage WHERE subject = ? AND day = ?", (subject, cat_today()))


store = UserStore()
