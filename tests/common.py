"""Shared helpers for sandbox tests. Import first in every test."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "tools"))

# Selection data ends here. Everything after is the locked holdout: look at it
# ONCE per idea, only after the config is written in TEST_LOG.md. Never tune on it.
LOCK_DATE = "2025-08-29"
