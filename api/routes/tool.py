# tool.py

import sys
import os
import logging
import importlib
import inspect
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger("tool_routes")

# Setup safe path handling
try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    logger.debug(f"Path configuration: {sys.path}")
except Exception as e:
    logger.error(f"Failed to configure import paths: {e}")

# Safely import ToolDiscovery with fallback
try:
    from langchain_tools.tool_discovery import ToolDiscovery
except ImportError:
    logger.error("Failed to import ToolDiscovery. Using minimal implementation.")
    
    # Minimal ToolDiscovery implementation as fallback
    class ToolDiscovery:
        def __init__(self, tools_dir=None):
            self.tools_dir = tools_dir or "langchain_tools"
            
        def list_tools(self):
            try:
                if not os.path.isdir(self.tools_dir):
                    logger.warning(f"Tools directory not found: {self.tools_dir}")
                    return []
                
                return [f[:-3] for f in os.listdir(self.tools_dir) 
                        if f.endswith('.py') and not f.startswith('_') 
                        and f != "__init__.py"]
            except Exception as e:
                logger.error(f"Failed to list tools: {e}")
                return []

# Router Setup
tool_router = APIRouter()

# Tool Execution Model
class ToolRequest(BaseModel):
    tool: str
    input: dict

# Tool Execution Handler with robust error handling
@tool_router.post("/tool/execute", status_code=status.HTTP_200_OK)
def execute_tool(request: ToolRequest):
    tool_name = request.tool
    try:
        try:
            module = importlib.import_module(f"langchain_tools.{tool_name}")
        except ImportError:
            module = importlib.import_module(f"langchain_tools.{tool_name}_tool")

        func = getattr(module, tool_name, None)
        if not callable(func):
            return {"status": "error", "tool": tool_name, "message": f"No callable '{tool_name}' found."}
        result = func(**request.input)
        return {"status": "success", "tool": tool_name, "input": request.input, "output": result}
    except Exception as e:
        logger.warning(f"[ToolRouter] Tool '{tool_name}' execution failed: {e}")
        return {"status": "error", "tool": tool_name, "message": str(e)}

# Metadata Extractor with enhanced reliability
def get_tool_metadata(tool_name: str) -> dict:
    """
    Safely gets tool metadata without crashing on import errors.
    Tries both direct and _tool suffix for file/module import.
    """
    metadata = {
        "name": tool_name,
        "doc": "Unknown",
        "params": []
    }

    try:
        try:
            module = importlib.import_module(f"langchain_tools.{tool_name}")
        except ImportError:
            module = importlib.import_module(f"langchain_tools.{tool_name}_tool")

        func = getattr(module, tool_name, None)
        if not callable(func):
            metadata["doc"] = f"Error: '{tool_name}' exists but is not callable"
            return metadata

        metadata["doc"] = inspect.getdoc(func) or "No description provided"
        sig = inspect.signature(func)
        metadata["params"] = list(sig.parameters.keys())

    except Exception as e:
        logger.error(f"[ToolMetadata] Failed to load metadata for tool '{tool_name}': {e}")
        metadata["doc"] = f"Error: {str(e)}"

    return metadata

# Tool List Endpoint with improved error handling
@tool_router.get("/tool/list", status_code=status.HTTP_200_OK)
def list_all_tools():
    try:
        discovery = ToolDiscovery(tools_dir="langchain_tools")
        tool_names = discovery.list_tools()
        
        if not tool_names:
            logger.warning("[ToolRouter] No tools found in primary directory, trying alternatives")
            
            alt_dirs = [
                os.path.join(base_dir, "langchain_tools"),
                "tools",
                os.path.join(base_dir, "tools")
            ]
            
            for alt_dir in alt_dirs:
                if os.path.isdir(alt_dir):
                    discovery = ToolDiscovery(tools_dir=alt_dir)
                    tool_names = discovery.list_tools()
                    if tool_names:
                        logger.info(f"[ToolRouter] Found tools in alternate directory: {alt_dir}")
                        break

        if not tool_names:
            logger.warning("[ToolRouter] No tools found in any directory")
            return {"tools": [], "message": "No tools found."}

        tools_metadata = []
        for name in tool_names:
            try:
                metadata = get_tool_metadata(name)
                tools_metadata.append(metadata)
            except Exception as e:
                logger.error(f"[ToolRouter] Failed to get metadata for tool '{name}': {e}")
                tools_metadata.append({
                    "name": name,
                    "doc": f"Error getting metadata: {str(e)}",
                    "params": []
                })

        return {"tools": tools_metadata}
    except Exception as e:
        logger.error(f"[ToolRouter] Error in list_all_tools: {e}")
        return {"tools": [], "error": str(e)}

# ✅ NEW: Tool Schema Discovery Endpoint
@tool_router.get("/tool/discover/{tool_name}", status_code=status.HTTP_200_OK)
def discover_tool_schema(tool_name: str):
    """
    Returns detailed metadata (doc, param schema) for a given tool.
    """
    try:
        metadata = get_tool_metadata(tool_name)
        if not metadata.get("doc") or metadata["doc"].startswith("Error"):
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found or broken.")
        return {"status": "success", "tool": tool_name, "schema": metadata}
    except Exception as e:
        logger.error(f"[ToolRouter] Failed to discover schema for '{tool_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Discovery error: {e}")
