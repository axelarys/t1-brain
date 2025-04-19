#!/usr/bin/env python3
"""
Debug script to identify import and module loading issues.
Run this from the project root to diagnose tool loading problems.
"""

import os
import sys
import logging
import importlib
import traceback

# Configure logging to show detailed information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_tools")

def check_environment():
    """Check Python environment and system paths"""
    logger.info("=== Python Environment Information ===")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Current Working Directory: {os.getcwd()}")
    logger.info(f"Module Search Paths:")
    for i, path in enumerate(sys.path):
        logger.info(f"  {i}: {path}")
    
    # Check for project directories
    important_dirs = [
        "langchain_tools",
        "tools",
        "api",
        "api/routes"
    ]
    
    logger.info("=== Project Directory Check ===")
    for dir_path in important_dirs:
        exists = os.path.isdir(dir_path)
        logger.info(f"Directory '{dir_path}': {'EXISTS' if exists else 'MISSING'}")
        if exists:
            try:
                files = os.listdir(dir_path)
                logger.info(f"  Contains {len(files)} files: {', '.join(files[:5])}" + 
                           (f" (and {len(files)-5} more)" if len(files) > 5 else ""))
            except Exception as e:
                logger.error(f"  Error reading directory: {e}")

def try_import_tool_discovery():
    """Try to import ToolDiscovery from different possible locations"""
    logger.info("=== Testing ToolDiscovery Import ===")
    
    import_paths = [
        "langchain_tools.tool_discovery",
        "tools.tool_discovery",
        "tool_discovery"
    ]
    
    for path in import_paths:
        try:
            logger.info(f"Trying to import from: {path}")
            module = importlib.import_module(path)
            logger.info(f"SUCCESS! Module imported from {path}")
            
            # Check for ToolDiscovery class
            if hasattr(module, "ToolDiscovery"):
                logger.info("ToolDiscovery class found in module")
                
                # Try to instantiate and call list_tools
                try:
                    discovery = module.ToolDiscovery()
                    tools = discovery.list_tools()
                    logger.info(f"list_tools() returned: {tools}")
                except Exception as e:
                    logger.error(f"Error instantiating or using ToolDiscovery: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.error("Module found but ToolDiscovery class is missing")
                
            return True
        except ImportError as e:
            logger.warning(f"Failed to import from {path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error importing from {path}: {e}")
            logger.error(traceback.format_exc())
    
    logger.error("All import attempts failed!")
    return False

def test_tool_imports():
    """Test importing all tool modules"""
    logger.info("=== Testing Tool Module Imports ===")
    
    # First locate the tools directory
    potential_dirs = ["langchain_tools", "tools"]
    tools_dir = None
    
    for dir_path in potential_dirs:
        if os.path.isdir(dir_path):
            logger.info(f"Found tools directory: {dir_path}")
            tools_dir = dir_path
            break
    
    if not tools_dir:
        logger.error("No valid tools directory found!")
        return
    
    # Get list of potential tool modules
    try:
        tool_files = [f[:-3] for f in os.listdir(tools_dir) 
                    if f.endswith('.py') and not f.startswith('_')
                    and f != "__init__.py" and f != "tool_discovery.py"]
        
        logger.info(f"Found {len(tool_files)} potential tool modules: {tool_files}")
        
        # Try to import each one
        for tool_name in tool_files:
            logger.info(f"\nTesting import for tool: {tool_name}")
            full_module_name = f"{tools_dir}.{tool_name}"
            
            try:
                module = importlib.import_module(full_module_name)
                logger.info(f"Successfully imported module {full_module_name}")
                
                # Check if the module has the expected function
                if hasattr(module, tool_name):
                    logger.info(f"  Tool function '{tool_name}' found in module")
                    
                    # Check if it's callable
                    if callable(getattr(module, tool_name)):
                        logger.info(f"  Tool function '{tool_name}' is callable")
                    else:
                        logger.error(f"  ISSUE: '{tool_name}' exists but is not callable!")
                else:
                    logger.error(f"  ISSUE: Module missing expected function '{tool_name}'")
                    
            except Exception as e:
                logger.error(f"Failed to import {full_module_name}: {e}")
                logger.error(traceback.format_exc())
                
    except Exception as e:
        logger.error(f"Error listing tool files: {e}")
        logger.error(traceback.format_exc())

def simulate_tool_route():
    """Simulate the tool router's loading process"""
    logger.info("=== Simulating Tool Route Loading ===")
    
    try:
        # Add proper path handling similar to what should be in tool.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        
        # Attempt to import tool_discovery
        try:
            from langchain_tools.tool_discovery import ToolDiscovery
            logger.info("Successfully imported ToolDiscovery")
            
            # Try to use it
            discovery = ToolDiscovery(tools_dir="langchain_tools")
            tool_names = discovery.list_tools()
            logger.info(f"Tool discovery found: {tool_names}")
            
        except ImportError as e:
            logger.error(f"Failed to import ToolDiscovery: {e}")
        
    except Exception as e:
        logger.error(f"Error in simulating tool route: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TOOL IMPORT DIAGNOSTIC SCRIPT")
    print("="*50 + "\n")
    
    check_environment()
    print("\n" + "-"*50 + "\n")
    
    try_import_tool_discovery()
    print("\n" + "-"*50 + "\n")
    
    test_tool_imports()
    print("\n" + "-"*50 + "\n")
    
    simulate_tool_route()
    print("\n" + "-"*50 + "\n")
    
    print("Diagnostic completed. Check the logs for issues.")