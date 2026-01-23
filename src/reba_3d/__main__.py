"""
Entry point for running reba_3d as a module.

Usage: python -m reba_3d [command] [options]
"""

import sys
from reba_3d.cli import main

if __name__ == "__main__":
    sys.exit(main())
