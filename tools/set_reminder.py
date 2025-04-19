# tools/set_reminder.py

import logging
from datetime import datetime, timedelta

logger = logging.getLogger("set_reminder")

class set_reminder:
    name = "set_reminder"
    description = "Sets a user reminder with a message and optional future timestamp."

    @staticmethod
    def run(input: dict) -> dict:
        """
        Expected input:
        {
            "message": "Call mom",
            "time": "in 2 hours" or "2024-05-01T14:00:00"
        }
        """
        try:
            message = input.get("message")
            time_str = input.get("time")

            if not message:
                return {
                    "status": "error",
                    "tool": "set_reminder",
                    "error": "Missing 'message' field in input"
                }

            # Simple time parsing fallback
            if not time_str:
                reminder_time = datetime.now() + timedelta(hours=1)
            else:
                try:
                    reminder_time = datetime.fromisoformat(time_str)
                except Exception:
                    reminder_time = datetime.now() + timedelta(hours=1)

            logger.info(f"[set_reminder] Reminder set: '{message}' at {reminder_time}")

            return {
                "status": "success",
                "tool": "set_reminder",
                "input": input,
                "output": f"Reminder set for '{message}' at {reminder_time.strftime('%Y-%m-%d %H:%M:%S')}"
            }

        except Exception as e:
            logger.error(f"[set_reminder] Failed: {e}")
            return {
                "status": "error",
                "tool": "set_reminder",
                "error": str(e)
            }