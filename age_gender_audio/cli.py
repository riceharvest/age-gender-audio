"""CLI entry point for age-gender-audio."""
import argparse

from . import __version__, greet


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="age-gender-audio", description="age-gender-audio CLI")
    p.add_argument("name", nargs="?", default="world")
    p.add_argument("--version", action="version", version=__version__)
    a = p.parse_args(argv)
    print(greet(a.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())