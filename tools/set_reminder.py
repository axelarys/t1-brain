# tools/set_reminder.py

def run_action(task: str, date: str = "", time: str = "", **kwargs) -> str:
    if not task:
        return "No task provided."

    reminder = f"📅 Reminder: {task}"
    if date:
        reminder += f" on {date}"
    if time:
        reminder += f" at {time}"

    return reminder
