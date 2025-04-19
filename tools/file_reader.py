# tools/file_reader.py

import os
import logging

logger = logging.getLogger("file_reader")

class file_reader:
    name = "file_reader"
    description = "Reads the contents of a text file given a valid absolute path."

    @staticmethod
    def run(input: dict) -> dict:
        """
        Expected input:
        {
            "path": "/absolute/path/to/file.txt"
        }
        """
        try:
            path = input.get("path")
            if not path or not os.path.isfile(path):
                return {
                    "status": "error",
                    "tool": "file_reader",
                    "error": f"Invalid or missing file path: {path}"
                }

            with open(path, "r") as f:
                content = f.read()

            logger.info(f"[file_reader] Read content from: {path}")

            return {
                "status": "success",
                "tool": "file_reader",
                "input": input,
                "output": content[:1000]  # Limit for safety
            }

        except Exception as e:
            logger.error(f"[file_reader] Failed to read file: {e}")
            return {
                "status": "error",
                "tool": "file_reader",
                "error": str(e)
            }