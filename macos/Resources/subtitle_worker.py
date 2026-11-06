"""App adapter: use the installed engine without loading any .env file."""
from subtitle_engine.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
