# utils/tool_discovery.py

import os
import importlib
import inspect
import logging
import sys

from memory.session_memory import PersistentSessionMemory

TOOL_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
TOOL_DIR = os.path.abspath(TOOL_DIR)
sys.path.append(TOOL_DIR)

logger = logging.getLogger("tool_discovery")
memory = PersistentSessionMemory()

def discover_tools() -> list[dict]:
    discovered = []
    try:
        for filename in os.listdir(TOOL_DIR):
            if filename.startswith("_") or not filename.endswith(".py"):
                continue

            module_name = filename[:-3]
            try:
                module_path = f"tools.{module_name}"
                if module_path in sys.modules:
                    del sys.modules[module_path]
                module = importlib.import_module(module_path)

                if hasattr(module, "run_action") and inspect.isfunction(module.run_action):
                    doc = inspect.getdoc(module.run_action) or "No description."
                    tool_info = {
                        "name": module_name,
                        "description": doc.strip()
                    }
                    discovered.append(tool_info)

                    # ✅ Register to memory (Redis)
                    memory.store_tool_metadata(module_name, doc.strip())

            except Exception as e:
                logger.warning(f"[ToolDiscovery] Skipped '{module_name}': {e}")
                continue

    except Exception as e:
        logger.error(f"[ToolDiscovery] Failed to scan tools: {e}")
        return []

    return sorted(discovered, key=lambda x: x["name"])
