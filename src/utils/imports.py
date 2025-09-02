"""Centralized imports for the project"""
import os
import sys
import logging

def setup_python_path():
    """Add project root to Python path"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
        logging.info(f"Added {project_root} to Python path")
