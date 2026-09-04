import Foundation

private final class LaTeXFontsBundleFinder {}

/// Where the KaTeX font bundle actually lives in a hand-assembled .app.
///
/// SwiftPM generates SwaTexRender's `Bundle.module` as
/// `Bundle(path: Bundle.main.bundleURL + "SwaTex_SwaTexRender.bundle")` with a
/// build-directory fallback, and traps when neither exists. For an app bundle
/// `bundleURL` is the .app itself — a location `codesign` refuses to seal
/// ("unsealed contents present in the bundle root"), so the fonts can only ship
/// in Contents/Resources and both the patched dependency (see
/// `scripts/patch-swatex-font-lookup.sh`) and this type search for them.
enum LaTeXFonts {
    static let bundleName = "SwaTex_SwaTexRender.bundle"
    /// One font that must be inside the bundle: an empty directory of the right
    /// name is a half-finished copy, not a hit. Relative to the bundle root —
    /// see `probeSubpaths` for why that is not the only place it sits.
    static let probeFont = "Fonts/KaTeX_Main-Regular.ttf"

    /// The two bundle layouts that ship the same fonts. SwiftPM's classic build
    /// directory writes a FLAT bundle (`Fonts/` at the root); the toolchain's
    /// swift-build layout writes a STRUCTURED macOS bundle
    /// (`Contents/Resources/Fonts/`), and `build.sh` copies whichever one
    /// `--show-bin-path` produced into the .app verbatim. `Bundle`'s own
    /// resource lookup — what actually LOADS a font, see
    /// `scripts/patch-swatex-font-lookup.sh` — reads both, so a probe that
    /// knows only the flat layout answers "no fonts" while the fonts sit right
    /// there, and every equation renders as its own source text.
    static let probeSubpaths = [probeFont, "Contents/Resources/" + probeFont]

    /// Contents/Resources for a real .app; the bundle URL covers the
    /// `swift build` layout, where the resource bundle sits beside the binary;
    /// its parent covers `swift test`, where the reading bundle is the
    /// .xctest and the resource bundle is its sibling.
    static func searchLocations(resourceURL: URL?, bundleURL: URL) -> [URL] {
        [resourceURL, bundleURL, bundleURL.deletingLastPathComponent()].compactMap { $0 }
    }

    static func locate(
        searching candidates: [URL],
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) -> URL? {
        for base in candidates {
            let bundle = base.appendingPathComponent(bundleName)
            for probe in probeSubpaths {
                if fileExists(bundle.appendingPathComponent(probe)) { return bundle }
            }
        }
        return nil
    }

    /// Resolved once. False means every LaTeX segment renders as its own source
    /// text — the app must never trap on a missing resource.
    static let isAvailable: Bool = {
        let own = Bundle(for: LaTeXFontsBundleFinder.self)
        let candidates = searchLocations(resourceURL: Bundle.main.resourceURL, bundleURL: Bundle.main.bundleURL)
            + searchLocations(resourceURL: own.resourceURL, bundleURL: own.bundleURL)
        return locate(searching: candidates) != nil
    }()
}
