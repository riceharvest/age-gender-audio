"""Smoke tests — the CI runs exactly these (python -m pytest -q)."""
import subprocess
import sys

import age_gender_audio
from age_gender_audio.cli import main


def test_version():
    assert age_gender_audio.__version__ == "0.1.0"


def test_greet():
    assert age_gender_audio.greet("world") == "hello, world!"


def test_cli(capsys):
    assert main(["lab"]) == 0
    assert capsys.readouterr().out.strip() == "hello, lab!"


def test_cli_module():
    r = subprocess.run([sys.executable, "-m", "age_gender_audio.cli", "ci"], capture_output=True, text=True)
    assert r.returncode == 0
    assert r.stdout.strip() == "hello, ci!"