#!/bin/zsh
set -eu
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/../.."
xcodegen generate --spec macos/project.yml
xcodebuild -project macos/SubtitleRefine.xcodeproj -scheme SubtitleRefine \
  -configuration Debug -derivedDataPath macos/.build \
  -destination 'platform=macOS,arch=arm64' CODE_SIGN_IDENTITY=- build
macos/.build/Build/Products/Debug/SubtitleRefineProbe snapshots macos/.build/snapshots
