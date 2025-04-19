# utils/audit_logger.py

import os
import json
from datetime import datetime

AUDIT_LOG_FILE = "/root/projects/t1-brain/logs/audit.log"

def log_audit_event(api_key: str, endpoint: str, action: str, session_id: str, status: str):
    """
    Logs a structured audit event to the audit.log file.
    """
    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)

    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_key": api_key,
        "endpoint": endpoint,
        "action": action,
        "session_id": session_id,
        "status": status
    }

    try:
        with open(AUDIT_LOG_FILE, "a") as log_file:
            log_file.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"[AuditLogger] Failed to write audit log: {e}")
