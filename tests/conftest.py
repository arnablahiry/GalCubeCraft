import os
import sys

# Ensure the repository `src/` directory is on sys.path so imports like
# `import GalCubeCraft` work in CI and local pytest runs even when the package
# is not installed into the environment.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
