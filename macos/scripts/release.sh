#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/../.."
xcodegen generate --spec macos/project.yml
xcodebuild -project macos/SubtitleRefine.xcodeproj -scheme SubtitleRefine \
  -configuration Release -derivedDataPath macos/.build/release \
  -destination 'platform=macOS,arch=arm64' CODE_SIGN_IDENTITY=- build
mkdir -p macos/dist
# Only replace this script's own generated deliverables.
rm -rf macos/dist/SubtitleRefine.app
/usr/bin/ditto macos/.build/release/Build/Products/Release/SubtitleRefine.app macos/dist/SubtitleRefine.app
/usr/bin/codesign --force --sign - macos/dist/SubtitleRefine.app
/usr/bin/codesign --verify --deep --strict --verbose=2 macos/dist/SubtitleRefine.app
icon_file=$(/usr/libexec/PlistBuddy -c 'Print CFBundleIconFile' macos/dist/SubtitleRefine.app/Contents/Info.plist)
[[ "$icon_file" == "AppIcon.icns" ]]
[[ -s "macos/dist/SubtitleRefine.app/Contents/Resources/$icon_file" ]]
cmp macos/Resources/subtitle_worker.py macos/dist/SubtitleRefine.app/Contents/Resources/subtitle_worker.py
rm -f macos/dist/SubtitleRefine-macOS-arm64.zip
/usr/bin/ditto -c -k --sequesterRsrc --keepParent macos/dist/SubtitleRefine.app macos/dist/SubtitleRefine-macOS-arm64.zip
/usr/bin/shasum -a 256 macos/dist/SubtitleRefine-macOS-arm64.zip > macos/dist/SHA256SUMS
printf 'Release ready: macos/dist/SubtitleRefine.app and SubtitleRefine-macOS-arm64.zip\n'
