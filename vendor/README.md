# Bundled subtitle-engine release

- Distribution: subtitle-engine 0.1.1 (MIT; license included in wheel).
- Source repository: private `b12031106/subtitle-engine`, commit `39ac006`.
  This wheel makes this consumer
  commit installable without GitHub access or reliance on a sibling directory.
- SHA256SUMS fixes the exact shipped bytes. `python scripts/install_engine.py` verifies and
  installs with that interpreter, without dependency downloads. Original provider/media
  dependencies remain in requirements.txt.
- All package .py entries were compared byte-for-byte with the engine source before release.
- Rebuild from that engine checkout with
  `SOURCE_DATE_EPOCH=1792771200 python -m pip wheel --no-deps --wheel-dir dist .`.
  Setuptools/build-tool changes may alter wheel metadata; use the checked-in artifact for
  exact repeat installations. Do not edit the wheel or maintain a second source copy here.
- Before a future update, change the engine version and rebuild/test; update wheel filename,
  SHA256SUMS, requirements and this provenance together. Keep the running worker unchanged
  until current tasks drain and the new dependency is installed and verified.
