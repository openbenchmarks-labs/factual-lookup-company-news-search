from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")
