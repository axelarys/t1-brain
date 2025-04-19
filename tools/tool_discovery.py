import os
import sys
import logging
import importlib
import inspect
import json
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("tool_discovery")

class ToolDiscovery:
    """
    Enhanced tool discovery with caching, error recovery, and graceful failure handling.
    """
    # Class-level cache to avoid redundant scanning
    _tool_cache = {}
    _cache_timestamp = 0
    _cache_validity = 300  # Cache valid for 5 minutes

    def __init__(self, tools_dir=None):
        """
        Initialize with path to tools directory.
        If none provided, searches in multiple common locations.
        """
        self.tools_dir = None
        self.cached_tools = []
        
        # Try to find tools directory
        if tools_dir:
            # Use provided directory if it exists
            if os.path.isdir(tools_dir):
                self.tools_dir = os.path.abspath(tools_dir)
            else:
                logger.warning(f"Provided tools directory not found: {tools_dir}")
        
        # Auto-discover if needed
        if not self.tools_dir:
            self._auto_discover_tools_dir()
            
        logger.debug(f"[ToolDiscovery] Initialized with tools directory: {self.tools_dir}")
        
        # Try to load from cache first
        self._load_from_cache()

    def _auto_discover_tools_dir(self):
        """Auto-discover tools directory from common locations."""
        # Common locations to check
        base_path = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            os.path.join(base_path, "langchain_tools"),
            os.path.join(base_path, "tools"),
            "langchain_tools",
            "tools",
            os.path.join(os.path.dirname(base_path), "langchain_tools"),
            os.path.join(os.path.dirname(base_path), "tools"),
        ]
        
        # Check for first existing directory
        for path in potential_paths:
            if os.path.isdir(path):
                self.tools_dir = os.path.abspath(path)
                logger.info(f"[ToolDiscovery] Auto-discovered tools directory: {self.tools_dir}")
                return
                
        # Fallback to current directory
        logger.warning("[ToolDiscovery] No tools directory found, using current directory")
        self.tools_dir = os.path.abspath(os.getcwd())

    def _is_valid_tool_file(self, filename: str) -> bool:
        """Check if a file is a valid Python module that could be a tool."""
        return (
            filename.endswith('.py') and 
            not filename.startswith('_') and 
            filename != "__init__.py" and
            filename != "tool_discovery.py"
        )
    
    def _load_from_cache(self) -> bool:
        """Load tools from class cache if valid."""
        cache_key = self.tools_dir
        
        # Check if we have a valid cache
        current_time = time.time()
        if (cache_key in self._tool_cache and 
            (current_time - self._cache_timestamp) < self._cache_validity):
            
            self.cached_tools = self._tool_cache[cache_key]
            logger.debug(f"[ToolDiscovery] Loaded {len(self.cached_tools)} tools from cache")
            return True
            
        return False
        
    def _update_cache(self, tools: List[str]):
        """Update the class-level cache with discovered tools."""
        self._tool_cache[self.tools_dir] = tools
        self._cache_timestamp = time.time()

    def list_tools(self) -> List[str]:
        """
        Lists all valid tool modules in the tools directory.
        Uses cached results if available and recent.
        
        Returns:
            list[str]: List of valid tool names
        """
        # Return cached tools if available
        if self.cached_tools:
            return self.cached_tools
            
        tool_names = []
        
        # Verify directory exists
        if not self.tools_dir or not os.path.isdir(self.tools_dir):
            logger.warning(f"[ToolDiscovery] Tools directory not found: {self.tools_dir}")
            return tool_names
            
        try:
            # Get all potential tool files
            for file in os.listdir(self.tools_dir):
                if self._is_valid_tool_file(file):
                    tool_names.append(file[:-3])  # remove .py extension
            
            logger.debug(f"[ToolDiscovery] Found {len(tool_names)} tools: {tool_names}")
            
            # Update cache
            self._update_cache(tool_names)
            self.cached_tools = tool_names
            
        except Exception as e:
            logger.error(f"[ToolDiscovery] Failed to list tools: {e}")
            
        return tool_names
        
    def get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """
        Gets metadata for a specific tool safely.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            dict: Tool metadata including name, doc, and parameters
        """
        metadata = {
            "name": tool_name,
            "doc": "Unknown",
            "params": [],
            "status": "unknown"
        }
        
        if not tool_name:
            metadata["status"] = "error"
            metadata["doc"] = "No tool name provided"
            return metadata
            
        try:
            # Ensure tools directory is on the path
            if self.tools_dir and os.path.isdir(self.tools_dir):
                tools_parent = os.path.dirname(self.tools_dir)
                if tools_parent not in sys.path:
                    sys.path.insert(0, tools_parent)
            
            # Get module name
            module_path = f"langchain_tools.{tool_name}"
            if self.tools_dir and os.path.basename(self.tools_dir) != "langchain_tools":
                module_path = f"{os.path.basename(self.tools_dir)}.{tool_name}"
                
            # Try to import module
            try:
                module = importlib.import_module(module_path)
                metadata["status"] = "module_found"
            except ImportError as e:
                metadata["status"] = "import_failed"
                metadata["doc"] = f"Error importing tool: {str(e)}"
                return metadata
                
            # Get function and extract metadata
            func = getattr(module, tool_name, None)
            if not callable(func):
                metadata["status"] = "not_callable"
                metadata["doc"] = f"Error: '{tool_name}' exists but is not callable"
                return metadata
                
            # Extract documentation and parameters
            metadata["doc"] = inspect.getdoc(func) or "No description provided"
            sig = inspect.signature(func)
            metadata["params"] = list(sig.parameters.keys())
            metadata["status"] = "success"
            
        except Exception as e:
            metadata["status"] = "error"
            metadata["doc"] = f"Error extracting metadata: {str(e)}"
            
        return metadata
        
    def list_tools_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Lists all tools with their metadata.
        
        Returns:
            list: List of tool metadata dictionaries
        """
        tool_names = self.list_tools()
        
        # Get metadata for each tool
        tools_with_metadata = []
        for name in tool_names:
            try:
                metadata = self.get_tool_metadata(name)
                tools_with_metadata.append(metadata)
            except Exception as e:
                logger.error(f"[ToolDiscovery] Failed to get metadata for {name}: {e}")
                tools_with_metadata.append({
                    "name": name,
                    "doc": f"Error getting metadata: {str(e)}",
                    "params": [],
                    "status": "error"
                })
                
        return tools_with_metadata
        
    def save_tools_cache(self, cache_file="tools_cache.json") -> bool:
        """
        Save tool information to a cache file.
        
        Args:
            cache_file: Path to save the cache
            
        Returns:
            bool: Success or failure
        """
        try:
            tools = self.list_tools_with_metadata()
            
            cache_data = {
                "timestamp": time.time(),
                "tools_dir": self.tools_dir,
                "tools": tools
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"[ToolDiscovery] Saved {len(tools)} tools to cache file")
            return True
            
        except Exception as e:
            logger.error(f"[ToolDiscovery] Failed to save tools cache: {e}")
            return False
            
    def load_tools_cache(self, cache_file="tools_cache.json") -> bool:
        """
        Load tool information from a cache file.
        
        Args:
            cache_file: Path to the cache file
            
        Returns:
            bool: Success or failure
        """
        try:
            if not os.path.exists(cache_file):
                logger.warning(f"[ToolDiscovery] Cache file not found: {cache_file}")
                return False
                
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is for the same directory
            if cache_data.get("tools_dir") != self.tools_dir:
                logger.warning("[ToolDiscovery] Cache is for a different tools directory")
                return False
                
            # Check cache freshness (24 hour validity)
            if (time.time() - cache_data.get("timestamp", 0)) > 86400:
                logger.warning("[ToolDiscovery] Cache is more than 24 hours old")
                return False
                
            # Update cache
            self._tool_cache[self.tools_dir] = [t["name"] for t in cache_data["tools"]]
            self._cache_timestamp = cache_data.get("timestamp", time.time())
            self.cached_tools = self._tool_cache[self.tools_dir]
            
            logger.info(f"[ToolDiscovery] Loaded {len(self.cached_tools)} tools from cache file")
            return True
            
        except Exception as e:
            logger.error(f"[ToolDiscovery] Failed to load tools cache: {e}")
            return False