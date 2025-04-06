# tools/set_reminder.py

def run_action(task: str, date: str = "", time: str = "", **kwargs) -> str:
    """
    Set a natural language reminder.

    Parameters:
    - task (str): Description of the reminder.
    - date (str): Optional date string.
    - time (str): Optional time string.

    Returns:
    - str: Human-readable reminder message.
    """
    if not task:
        return "⚠️ No task provided."

    reminder = f"📅 Reminder: {task}"
    if date:
        reminder += f" on {date}"
    if time:
        reminder += f" at {time}"

    return reminder
