"""Allow the package to be run with ``python -m zoomer``."""

from zoomer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
