# Bundled subtitle-engine release

- Distribution: subtitle-engine 0.1.0 (MIT; license included in wheel).
- Source repository: subtitle-engine, engine commit `5a9527d31411a6211c9d857dc0a5162cbdcee767`.
- The engine repository currently exists locally and has no configured remote. This wheel
  makes this consumer commit installable without inventing a remote or relying on a sibling directory.
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
