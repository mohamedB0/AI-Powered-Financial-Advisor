"""
notebooks/_setup.py
Shared setup for all Jupyter notebooks in this project.

Usage (first cell of any notebook):
    from _setup import PROJECT_ROOT

This module:
  1. Finds the project root (parent of this file)
  2. Adds it to sys.path
  3. Changes CWD to project root
  4. Loads .env
"""
import sys
import os
from pathlib import Path

# This file lives in notebooks/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
