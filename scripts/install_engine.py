"""Install the bundled immutable engine with this interpreter (no network/deps)."""
import hashlib
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
checksum, filename = (root / 'vendor/SHA256SUMS').read_text().strip().split()
wheel = root / 'vendor' / filename
if hashlib.sha256(wheel.read_bytes()).hexdigest() != checksum:
    raise SystemExit('Bundled engine checksum mismatch')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheel)], check=True)
