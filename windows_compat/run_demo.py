# -*- coding: utf-8 -*-
"""
One-command entry point for the native-Windows port of demo.py.

    windows_compat\\.venv\\Scripts\\python.exe windows_compat\\run_demo.py

It injects the sage.all shim into sys.path, switches the working directory to
the repository root (demo.py uses relative ./data paths), and runs demo.py
unchanged.  No global PYTHONPATH / conda / WSL / SageMath needed.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, HERE)   # windows_compat/sage -> from sage.all import ...
sys.path.insert(0, ROOT)   # pycompod
os.chdir(ROOT)

sys.argv = [os.path.join(ROOT, "demo.py")]
runpy.run_path(sys.argv[0], run_name="__main__")
