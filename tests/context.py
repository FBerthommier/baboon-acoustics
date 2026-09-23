"""Shared test fixtures: repository paths and loaded example signals."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baboon_acoustics import load_audio  # noqa: E402

WAHOO4 = REPO_ROOT / "data" / "wahoo_4.wav"
BARK = REPO_ROOT / "data" / "The_Baboon_Bark.wav"


class WahooTestCase(unittest.TestCase):
    """Base class providing the wahoo_4 signal to test cases."""

    @classmethod
    def setUpClass(cls):
        cls.x, cls.fs = load_audio(WAHOO4)
