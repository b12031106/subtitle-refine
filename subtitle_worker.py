"""Compatibility entry point; implementation is supplied by subtitle-engine."""
import sys
from pathlib import Path
try:
    from subtitle_engine import worker as _implementation
except ImportError as error:
    raise SystemExit("Install the pinned engine first: python scripts/install_engine.py") from error

if __name__ == "__main__":
    # Preserve the old script-local .env lookup even when called by media-cellar.
    _implementation.load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
    from subtitle_engine.worker import main
    sys.exit(main())
else:
    # Preserve imports and patch targets used by existing integrations/tests.
    sys.modules[__name__] = _implementation
