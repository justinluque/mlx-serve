import XCTest
@testable import MLXCore

/// Pure helpers behind `generate_image`'s `model` argument and the
/// `list_image_models` tool: which id resolves to which preset, and how the
/// catalog reads back to the model.
final class ImageModelCatalogTests: XCTestCase {

    func testResolveFindsABuiltinById() {
        let match = ImageModelCatalog.resolve(requestedId: ImageModelPreset.flux2Klein4B_Q4.id) { _ in nil }
        XCTAssertEqual(match?.id, ImageModelPreset.flux2Klein4B_Q4.id)
    }

    func testResolveFallsBackToTheCustomLookup() {
        let custom = ImageModelPreset.flux2Klein9B_Q4 // stand-in preset shape
        let match = ImageModelCatalog.resolve(requestedId: "some/custom-repo") { id in
            id == "some/custom-repo" ? custom : nil
        }
        XCTAssertEqual(match?.id, custom.id)
    }

    func testResolveReturnsNilForAnUnknownId() {
        let match = ImageModelCatalog.resolve(requestedId: "not-a-real-model") { _ in nil }
        XCTAssertNil(match)
    }

    func testListingTextNamesTheDefaultAndUndownloadedBuiltins() {
        let text = ImageModelCatalog.listingText(
            customs: [], currentId: ImageModelPreset.flux2Klein4B_Q4.id,
            isDownloaded: { $0.id == ImageModelPreset.flux2Klein4B_Q4.id })

        XCTAssertTrue(text.contains("\(ImageModelPreset.flux2Klein4B_Q4.id) — \(ImageModelPreset.flux2Klein4B_Q4.name) (default)"))
        // Every OTHER built-in is reported not-downloaded under this closure.
        for preset in ImageModelPreset.all where preset.id != ImageModelPreset.flux2Klein4B_Q4.id {
            XCTAssertTrue(text.contains("\(preset.id) — \(preset.name) (not downloaded yet"),
                          "\(preset.id) should be marked not downloaded")
        }
    }

    func testListingTextIncludesCustomModelsAsAlwaysAvailable() {
        let custom = ImageModelPreset.flux2Klein9B_Q4
        let text = ImageModelCatalog.listingText(
            customs: [custom], currentId: "something-else", isDownloaded: { _ in true })
        // A custom row carries no "not downloaded" tag — it's on disk by construction.
        XCTAssertTrue(text.contains("- \(custom.id) — \(custom.name)\n")
                      || text.hasSuffix("- \(custom.id) — \(custom.name)"))
    }

    func testEveryBuiltinListedExactlyOnce() {
        let text = ImageModelCatalog.listingText(customs: [], currentId: "", isDownloaded: { _ in true })
        for preset in ImageModelPreset.all {
            XCTAssertEqual(text.components(separatedBy: "- \(preset.id) —").count - 1, 1)
        }
    }
}
