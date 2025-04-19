# test_tool_discovery.py
# ✅ CLI test runner for ToolDiscovery

import sys
import os
import logging

# Add root project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from tools directory
from tools.tool_discovery import ToolDiscovery

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_test():
    logger.info("🔍 Running Tool Discovery Test...")

    # Set correct tools directory
    discovery = ToolDiscovery(tools_dir="/root/projects/t1-brain/tools")
    tools = discovery.list_tools()

    logger.info(f"✅ Discovered {len(tools)} tool(s): {tools}")
    if not tools:
        logger.warning("⚠️ No tools found. Make sure your tools/ directory has valid .py files.")

if __name__ == "__main__":
    run_test()