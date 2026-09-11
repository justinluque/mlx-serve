import Foundation

/// Pure helpers behind the `generate_image` / `list_image_models` agent
/// tools: which id resolves to which preset, and how the catalog is
/// described back to the model. Kept separate from `ChatTurnEngine` (and its
/// live `AppState`/UserDefaults reads) so both are unit-testable without a
/// server or a saved settings blob.
enum ImageModelCatalog {
    /// `list_image_models`'s text: every built-in preset (marked `default`
    /// and/or `not downloaded yet` as it applies) plus every custom on-disk
    /// model (always downloaded by construction — discovered FROM `/v1/models`).
    /// `isDownloaded`/`currentId` are injected so this touches neither disk
    /// nor `UserDefaults` directly.
    static func listingText(
        customs: [ImageModelPreset], currentId: String,
        isDownloaded: (ImageModelPreset) -> Bool
    ) -> String {
        var lines = ["Available image models (pass the id as generate_image's \"model\" argument):"]
        for preset in ImageModelPreset.all {
            var tags: [String] = []
            if preset.id == currentId { tags.append("default") }
            if !isDownloaded(preset) { tags.append("not downloaded yet — open the Image window once to fetch it") }
            let suffix = tags.isEmpty ? "" : " (\(tags.joined(separator: ", ")))"
            lines.append("- \(preset.id) — \(preset.name)\(suffix)")
        }
        for preset in customs {
            let suffix = preset.id == currentId ? " (default)" : ""
            lines.append("- \(preset.id) — \(preset.name)\(suffix)")
        }
        return lines.joined(separator: "\n")
    }

    /// Resolve an explicit `generate_image` `model` argument against the
    /// catalog. nil when the id matches neither a built-in preset nor a
    /// custom on-disk model — the caller steers back to `list_image_models`
    /// rather than guessing, because a model an agent invents by guessing is
    /// a model that silently generates with the wrong preset.
    static func resolve(requestedId: String, customLookup: (String) -> ImageModelPreset?) -> ImageModelPreset? {
        ImageModelPreset.all.first { $0.id == requestedId } ?? customLookup(requestedId)
    }
}
