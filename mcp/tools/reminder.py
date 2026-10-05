import sqlite3
import os
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ReminderTool")
USER_TIMEZONE = os.getenv("USER_TIMEZONE", "Asia/Kolkata")
DEFAULT_PHONE = os.getenv("TARGET_PHONE_NUMBER", "")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "reminders.db")


def get_db_connection():
    """Returns a SQLite connection to reminders.db."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_reminder_db():
    """Creates the reminders table if it doesn't exist and migrates schema."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT NOT NULL,
                reminder_text TEXT NOT NULL,
                remind_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                recurrence TEXT DEFAULT 'none',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminders_pending ON reminders(status, remind_at)")
        
        # Migration: Add recurrence column if upgrading from an older schema
        try:
            cursor.execute("ALTER TABLE reminders ADD COLUMN recurrence TEXT DEFAULT 'none'")
        except sqlite3.OperationalError:
            pass  # Already exists

        conn.commit()


init_reminder_db()


def parse_and_normalize_time(time_str: str) -> Optional[datetime]:
    """
    Parses an ISO format or timestamp string and normalizes it to USER_TIMEZONE.
    """
    try:
        dt = datetime.fromisoformat(time_str.strip())
        tz = ZoneInfo(USER_TIMEZONE)
        if dt.tzinfo is None:
            # If naive, assume user's local timezone
            dt = dt.replace(tzinfo=tz)
        else:
            # Convert to user's local timezone
            dt = dt.astimezone(tz)
        return dt
    except Exception as e:
        logger.error(f"Failed to parse time string '{time_str}': {e}")
        return None


def schedule_reminder(
    remind_at_iso: str,
    reminder_text: str,
    recurrence: str = "none",
    phone_number: str = DEFAULT_PHONE
) -> str:
    """
    Schedules a new reminder (one-time, daily, or weekly) for the user.
    
    Args:
        remind_at_iso: Target date and time in ISO format (e.g. '2026-09-15T21:00:00')
        reminder_text: The reminder message to send (e.g. 'Call Mom')
        recurrence: 'none' (default), 'daily' (every day at same time), or 'weekly'
        phone_number: User's WhatsApp phone number
    """
    target_dt = parse_and_normalize_time(remind_at_iso)
    if not target_dt:
        return f"❌ Error: Invalid date/time format '{remind_at_iso}'. Please use ISO-8601 format (YYYY-MM-DDTHH:MM:SS)."

    tz = ZoneInfo(USER_TIMEZONE)
    now = datetime.now(tz)

    if target_dt <= now:
        # If scheduling for a recurring time today that has already passed, bump by 1 day
        clean_rec = recurrence.strip().lower() if recurrence else "none"
        if clean_rec == "daily":
            target_dt = target_dt + timedelta(days=1)
        elif clean_rec == "weekly":
            target_dt = target_dt + timedelta(days=7)
        else:
            return f"❌ Error: The scheduled time ({target_dt.strftime('%I:%M %p on %d %b %Y')}) is in the past! Current time is {now.strftime('%I:%M %p')}."

    clean_rec = recurrence.strip().lower() if recurrence else "none"
    if clean_rec not in ["none", "daily", "weekly"]:
        clean_rec = "none"

    iso_stored = target_dt.isoformat()
    clean_text = reminder_text.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reminders (phone_number, reminder_text, remind_at, recurrence, status)
            VALUES (?, ?, ?, ?, 'pending')
        """, (phone_number, clean_text, iso_stored, clean_rec))
        conn.commit()
        reminder_id = cursor.lastrowid

    formatted_time = target_dt.strftime("%A, %d %B at %I:%M %p (%Z)")
    
    if clean_rec == "daily":
        msg = f"✅ Recurring daily reminder #{reminder_id} successfully scheduled for *every day at {target_dt.strftime('%I:%M %p (%Z)')}*:\n\"{clean_text}\""
    elif clean_rec == "weekly":
        msg = f"✅ Recurring weekly reminder #{reminder_id} successfully scheduled for *every {target_dt.strftime('%A at %I:%M %p (%Z)')}*:\n\"{clean_text}\""
    else:
        msg = f"✅ Reminder #{reminder_id} successfully scheduled for *{formatted_time}*:\n\"{clean_text}\""

    logger.info(f"✅ Reminder #{reminder_id} scheduled ({clean_rec}) for {formatted_time}: '{clean_text}'")
    return msg


def list_reminders(phone_number: str = DEFAULT_PHONE) -> str:
    """
    Lists all active (pending) reminders for the user, including recurring ones.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, reminder_text, remind_at, recurrence, status
            FROM reminders
            WHERE phone_number = ? AND status = 'pending'
            ORDER BY remind_at ASC
        """, (phone_number,))
        rows = cursor.fetchall()

    if not rows:
        return "You have no active pending reminders."

    lines = ["📋 *Your Pending Reminders:*"]
    for row in rows:
        try:
            dt = datetime.fromisoformat(row["remind_at"])
            time_str = dt.strftime("%A, %d %b at %I:%M %p")
        except Exception:
            time_str = row["remind_at"]
        
        rec = row["recurrence"] if "recurrence" in row.keys() else "none"
        if rec == "daily":
            rec_badge = " *(🔁 Repeats Daily)*"
        elif rec == "weekly":
            rec_badge = " *(🔁 Repeats Weekly)*"
        else:
            rec_badge = ""

        lines.append(f"• *ID #{row['id']}* - {time_str}{rec_badge}: \"{row['reminder_text']}\"")

    return "\n".join(lines)


def cancel_reminder(reminder_id: int, phone_number: str = DEFAULT_PHONE) -> str:
    """
    Cancels a scheduled reminder by its ID.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE reminders
            SET status = 'cancelled'
            WHERE id = ? AND phone_number = ? AND status = 'pending'
        """, (reminder_id, phone_number))
        conn.commit()
        affected = cursor.rowcount

    if affected > 0:
        return f"✅ Reminder #{reminder_id} has been cancelled."
    else:
        return f"⚠️ No active pending reminder found with ID #{reminder_id}."


def get_due_reminders() -> List[Dict[str, Any]]:
    """
    Retrieves all pending reminders whose scheduled time has arrived.
    Called by the background scheduler daemon.
    """
    tz = ZoneInfo(USER_TIMEZONE)
    now_iso = datetime.now(tz).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, phone_number, reminder_text, remind_at, recurrence
            FROM reminders
            WHERE status = 'pending' AND remind_at <= ?
            ORDER BY remind_at ASC
        """, (now_iso,))
        rows = cursor.fetchall()

    return [dict(row) for row in rows]


def advance_reminder_recurrence(reminder_id: int, current_remind_at: str, recurrence: str):
    """
    Advances a recurring reminder to its next due date (e.g. +1 day for daily, +7 days for weekly).
    """
    try:
        dt = datetime.fromisoformat(current_remind_at)
        if recurrence == "daily":
            next_dt = dt + timedelta(days=1)
        elif recurrence == "weekly":
            next_dt = dt + timedelta(days=7)
        else:
            return mark_reminder_sent(reminder_id)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE reminders
                SET remind_at = ?
                WHERE id = ?
            """, (next_dt.isoformat(), reminder_id))
            conn.commit()
            logger.info(f"🔁 Advanced recurring reminder #{reminder_id} ({recurrence}) to {next_dt.isoformat()}")
    except Exception as e:
        logger.error(f"Error advancing reminder #{reminder_id}: {e}")
        mark_reminder_sent(reminder_id)


def mark_reminder_sent(reminder_id: int):
    """
    Marks a one-off reminder as sent so it won't be sent again.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE reminders
            SET status = 'sent'
            WHERE id = ?
        """, (reminder_id,))
        conn.commit()


# =====================================================================
# MCP Tool JSON Schemas for Hermes Agent (OpenAI Function Calling)
# =====================================================================

SCHEDULE_REMINDER_SCHEMA = {
    "type": "object",
    "properties": {
        "remind_at_iso": {
            "type": "string",
            "description": "The exact date and time when the reminder should first fire, in ISO-8601 format (e.g. '2026-09-15T21:00:00'). Calculate this using the current time provided in your system prompt."
        },
        "reminder_text": {
            "type": "string",
            "description": "The message or task to remind the user about (e.g. 'Call Mom')."
        },
        "recurrence": {
            "type": "string",
            "enum": ["none", "daily", "weekly"],
            "description": "Recurrence frequency: 'none' for one-time reminders, 'daily' for everyday reminders (e.g. 'every day at 9 PM'), or 'weekly' for once a week. Defaults to 'none'."
        }
    },
    "required": ["remind_at_iso", "reminder_text"],
}

LIST_REMINDERS_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}

CANCEL_REMINDER_SCHEMA = {
    "type": "object",
    "properties": {
        "reminder_id": {
            "type": "integer",
            "description": "The numerical ID of the reminder to cancel (e.g. 3)."
        }
    },
    "required": ["reminder_id"],
}
