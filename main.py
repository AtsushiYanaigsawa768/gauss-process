#!/usr/bin/env python3
"""
Entry point for the unified system identification pipeline.

Usage:
    python main.py input/*.mat --kernel rbf --out-dir gp_output
    python main.py --help

For detailed documentation, see src/pipeline/README.md
"""
import sys
from src.pipeline.unified_pipeline import main

if __name__ == '__main__':
    sys.exit(main())
