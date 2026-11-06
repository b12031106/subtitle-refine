#!/usr/bin/env swift
import AppKit
import Foundation

let root = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "macos")
let iconset = root.appendingPathComponent(".build/AppIcon.iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
func drawIcon(size: Int) -> Data {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    let transform = NSAffineTransform()
    transform.scale(by: CGFloat(size) / 1024)
    transform.concat()
    let tile = NSBezierPath(roundedRect: NSRect(x: 72, y: 72, width: 880, height: 880), xRadius: 196, yRadius: 196)
    NSGradient(starting: NSColor(calibratedRed: 0.04, green: 0.20, blue: 0.46, alpha: 1),
               ending: NSColor(calibratedRed: 0.06, green: 0.55, blue: 0.74, alpha: 1))!.draw(in: tile, angle: 60)
    NSColor.white.withAlphaComponent(0.16).setStroke()
    tile.lineWidth = 4; tile.stroke()
    let panel = NSBezierPath(roundedRect: NSRect(x: 210, y: 282, width: 604, height: 448), xRadius: 82, yRadius: 82)
    NSColor.white.withAlphaComponent(0.96).setFill(); panel.fill()
    let tail = NSBezierPath()
    tail.move(to: NSPoint(x: 305, y: 310)); tail.line(to: NSPoint(x: 305, y: 202))
    tail.line(to: NSPoint(x: 437, y: 308)); tail.close(); tail.fill()
    let ink = NSColor(calibratedRed: 0.06, green: 0.31, blue: 0.51, alpha: 1)
    for rect in [NSRect(x: 292, y: 564, width: 258, height: 42), NSRect(x: 578, y: 564, width: 154, height: 42),
                 NSRect(x: 292, y: 466, width: 158, height: 42), NSRect(x: 478, y: 466, width: 254, height: 42)] {
        ink.setFill(); NSBezierPath(roundedRect: rect, xRadius: 21, yRadius: 21).fill()
    }
    NSColor(calibratedRed: 0.08, green: 0.72, blue: 0.71, alpha: 1).setFill()
    NSBezierPath(roundedRect: NSRect(x: 292, y: 368, width: 300, height: 30), xRadius: 15, yRadius: 15).fill()
    let sparkle = NSBezierPath()
    sparkle.move(to: NSPoint(x: 794, y: 838)); sparkle.line(to: NSPoint(x: 819, y: 783))
    sparkle.line(to: NSPoint(x: 874, y: 758)); sparkle.line(to: NSPoint(x: 819, y: 733))
    sparkle.line(to: NSPoint(x: 794, y: 678)); sparkle.line(to: NSPoint(x: 769, y: 733))
    sparkle.line(to: NSPoint(x: 714, y: 758)); sparkle.line(to: NSPoint(x: 769, y: 783)); sparkle.close()
    NSColor(calibratedRed: 0.53, green: 0.98, blue: 0.91, alpha: 1).setFill(); sparkle.fill()
    NSGraphicsContext.restoreGraphicsState()
    return bitmap.representation(using: .png, properties: [:])!
}
for points in [16,32,128,256,512] {
    for scale in [1,2] {
        let name = "icon_\(points)x\(points)\(scale == 2 ? "@2x" : "").png"
        try drawIcon(size: points * scale).write(to: iconset.appendingPathComponent(name))
    }
}
try drawIcon(size: 1024).write(to: root.appendingPathComponent(".build/AppIcon-preview.png"))
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", root.appendingPathComponent("Resources/AppIcon.icns").path]
try process.run(); process.waitUntilExit()
guard process.terminationStatus == 0 else { fatalError("iconutil failed") }
print("Generated AppIcon.icns")
