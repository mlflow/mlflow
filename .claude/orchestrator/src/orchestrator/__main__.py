"""Allow `python -m orchestrator <PR>`."""

from orchestrator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
