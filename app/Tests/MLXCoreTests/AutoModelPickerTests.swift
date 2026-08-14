import XCTest
@testable import MLXCore

/// The composer no longer refuses a message just because nothing is loaded: it
/// reads what you're sending and picks a model for it. These pin the pure
/// decision — which model, and WHY — so the wiring in ChatView has nothing to
/// decide for itself.
final class AutoModelPickerTests: XCTestCase {

    private let gb = Int64(1_073_741_824)
    private var ram16: UInt64 { 16 * 1_073_741_824 }
    private var ram64: UInt64 { 64 * 1_073_741_824 }

    private func model(_ name: String,
                       modelType: String = "gemma4",
                       kind: ModelKind = .base,
                       vision: Bool = false,
                       ctx: Int? = 32768,
                       sizeGB: Double = 4) -> LocalModel {
        var m = LocalModel(
            id: "test:\(name)", name: name, path: "/models/\(name)",
            sizeFormatted: "\(sizeGB) GB", modelType: modelType, source: .mlxServe, kind: kind)
        m.hasVision = vision
        m.contextLength = ctx
        m.sizeBytes = Int64(sizeGB * Double(gb))
        return m
    }

    // MARK: - Nothing to pick

    func testNoChatModelsAtAllPicksNothing() {
        // The blocking gate sheet owns this case — it offers the starter
        // download. The picker must not invent a second answer for it.
        let media = model("ltx-2.5", modelType: "AudioVideo")
        let drafter = model("gemma4-12b-assistant", modelType: "gemma4_assistant", kind: .drafter)
        XCTAssertEqual(
            AutoModelPicker.pick(need: AutoModelNeed(), candidates: [media, drafter],
                                 physicalMemoryBytes: ram64, lastUsedPath: ""),
            .noneAvailable)
    }

    func testANonChatModelIsNeverPickedEvenWhenItWasLastUsed() {
        // `selectedModelPath` can point at anything; the picker only ever
        // returns something the server can serve chat completions from.
        let media = model("krea", modelType: "krea2_turbo")
        let chat = model("qwen3-8b-instruct")
        let pick = AutoModelPicker.pick(need: AutoModelNeed(), candidates: [media, chat],
                                        physicalMemoryBytes: ram64,
                                        lastUsedPath: media.path)
        XCTAssertEqual(pick, .use(path: chat.path, name: chat.name, reason: .ramFit))
    }

    // MARK: - Vision is a hard requirement

    func testAnAttachedImagePicksAModelThatCanSee() {
        let text = model("qwen3-8b-instruct", sizeGB: 8)
        let seer = model("gemma4-12b-it", vision: true, sizeGB: 7)
        let need = AutoModelNeed(needsVision: true)
        // Even with the text model last-used and larger, an image the model
        // cannot read makes the answer fiction.
        let pick = AutoModelPicker.pick(need: need, candidates: [text, seer],
                                        physicalMemoryBytes: ram64, lastUsedPath: text.path)
        XCTAssertEqual(pick, .use(path: seer.path, name: seer.name, reason: .vision))
    }

    func testNoVisionModelDownloadedOffersOneRatherThanAnsweringBlind() {
        let text = model("qwen3-8b-instruct")
        let pick = AutoModelPicker.pick(need: AutoModelNeed(needsVision: true),
                                        candidates: [text],
                                        physicalMemoryBytes: ram16, lastUsedPath: "")
        guard case let .download(repoId, _, reason) = pick else {
            return XCTFail("expected a download offer, got \(pick)")
        }
        XCTAssertEqual(reason, .vision)
        // The offer must be a model that can actually see — every Gemma 4 pick
        // ships SigLIP vision.
        XCTAssertTrue(repoId.lowercased().contains("gemma-4"), "\(repoId) isn't a vision pick")
    }

    func testTheOfferedVisionModelFitsTheMac() {
        let small = AutoModelPicker.pick(need: AutoModelNeed(needsVision: true), candidates: [],
                                         physicalMemoryBytes: 8 * 1_073_741_824, lastUsedPath: "")
        let big = AutoModelPicker.pick(need: AutoModelNeed(needsVision: true), candidates: [],
                                       physicalMemoryBytes: ram64, lastUsedPath: "")
        guard case let .download(smallRepo, _, _) = small,
              case let .download(bigRepo, _, _) = big else {
            return XCTFail("expected download offers")
        }
        XCTAssertNotEqual(smallRepo, bigRepo, "an 8 GB Mac and a 64 GB Mac get the same vision model")
    }

    // MARK: - Ranking

    func testTheModelYouUsedLastWins() {
        let a = model("qwen3-8b-instruct")
        let b = model("gemma4-12b-it", sizeGB: 7)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(), candidates: [a, b],
                                        physicalMemoryBytes: ram64, lastUsedPath: a.path)
        XCTAssertEqual(pick, .use(path: a.path, name: a.name, reason: .lastUsed))
    }

    func testWithNoHistoryTheBiggestModelTheMacCanRunWins() {
        let small = model("gemma4-e2b-it", sizeGB: 2)
        let mid = model("gemma4-12b-it", sizeGB: 7)
        let huge = model("deepseek-v4", modelType: "deepseek_v4", sizeGB: 120)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(), candidates: [small, mid, huge],
                                        physicalMemoryBytes: ram16, lastUsedPath: "")
        // 120 GB of weights on a 16 GB Mac is a load that fails, not a better answer.
        XCTAssertEqual(pick, .use(path: mid.path, name: mid.name, reason: .ramFit))
    }

    func testWhenNothingFitsTheSmallestIsOfferedRatherThanNothing() {
        // Refusing here would put the user back where this feature started.
        let big = model("gemma4-31b", sizeGB: 30)
        let bigger = model("laguna-s", sizeGB: 60)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(), candidates: [bigger, big],
                                        physicalMemoryBytes: 8 * 1_073_741_824, lastUsedPath: "")
        XCTAssertEqual(pick, .use(path: big.path, name: big.name, reason: .ramFit))
    }

    func testALastUsedModelTooBigForTheMacIsStillHonoured() {
        // The user picked it themselves once and it is what they expect back;
        // the RAM tier is our guess, not their instruction.
        let big = model("gemma4-31b", sizeGB: 30)
        let small = model("gemma4-e2b-it", sizeGB: 2)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(), candidates: [big, small],
                                        physicalMemoryBytes: ram16, lastUsedPath: big.path)
        XCTAssertEqual(pick, .use(path: big.path, name: big.name, reason: .lastUsed))
    }

    // MARK: - Context

    func testALongMessagePrefersAModelThatCanHoldIt() {
        let shortCtx = model("qwen3-8b-instruct", ctx: 8192, sizeGB: 8)
        let longCtx = model("gemma4-12b-it", ctx: 131_072, sizeGB: 7)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(estimatedTokens: 40_000),
                                        candidates: [shortCtx, longCtx],
                                        physicalMemoryBytes: ram64, lastUsedPath: shortCtx.path)
        XCTAssertEqual(pick, .use(path: longCtx.path, name: longCtx.name, reason: .longContext))
    }

    func testWhenNothingHoldsItTheLargestWindowIsPicked() {
        let a = model("a", ctx: 8192)
        let b = model("b", ctx: 32768)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(estimatedTokens: 500_000),
                                        candidates: [a, b],
                                        physicalMemoryBytes: ram64, lastUsedPath: "")
        XCTAssertEqual(pick, .use(path: b.path, name: b.name, reason: .longContext))
    }

    func testAnUnknownContextLengthIsNotTreatedAsZero() {
        // GGUF and configs we didn't parse carry nil. Excluding them would hide
        // every GGUF the user owns behind a "too long" verdict.
        let unknown = model("gguf-model", modelType: "gguf", ctx: nil)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(estimatedTokens: 3_000),
                                        candidates: [unknown],
                                        physicalMemoryBytes: ram64, lastUsedPath: "")
        XCTAssertEqual(pick, .use(path: unknown.path, name: unknown.name, reason: .ramFit))
    }

    // MARK: - Tools

    func testWithToolsOnAToolCallingModelWinsAmongEquals() {
        let plain = model("some-base-model")
        let toolish = model("qwen3-8b-instruct")
        XCTAssertTrue(toolish.hasToolCalling)
        XCTAssertFalse(plain.hasToolCalling)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(wantsTools: true),
                                        candidates: [plain, toolish],
                                        physicalMemoryBytes: ram64, lastUsedPath: "")
        XCTAssertEqual(pick, .use(path: toolish.path, name: toolish.name, reason: .tools))
    }

    func testToolsNeverOutrankVision() {
        // A capability the message REQUIRES beats one it merely benefits from.
        let toolish = model("qwen3-8b-instruct")
        let seer = model("gemma4-12b", vision: true)
        let pick = AutoModelPicker.pick(need: AutoModelNeed(needsVision: true, wantsTools: true),
                                        candidates: [toolish, seer],
                                        physicalMemoryBytes: ram64, lastUsedPath: "")
        XCTAssertEqual(pick, .use(path: seer.path, name: seer.name, reason: .vision))
    }

    // MARK: - Reading the composer

    func testAnAttachedImageMakesVisionARequirement() {
        let need = AutoModelNeed.from(text: "what is this?", imageCount: 1,
                                      pdfCharacters: 0, toolsEnabled: false)
        XCTAssertTrue(need.needsVision)
    }

    func testAPastedPdfCountsTowardTheContextEstimate() {
        let plain = AutoModelNeed.from(text: "summarise", imageCount: 0,
                                       pdfCharacters: 0, toolsEnabled: false)
        let withPdf = AutoModelNeed.from(text: "summarise", imageCount: 0,
                                         pdfCharacters: 400_000, toolsEnabled: false)
        XCTAssertGreaterThan(withPdf.estimatedTokens, plain.estimatedTokens + 50_000)
    }

    func testTheEstimateLeavesRoomForTheReply() {
        // A window that exactly fits the prompt has nowhere to put the answer.
        let need = AutoModelNeed.from(text: "hi", imageCount: 0, pdfCharacters: 0, toolsEnabled: false)
        XCTAssertGreaterThanOrEqual(need.estimatedTokens, 512)
    }

    // MARK: - Copy

    func testEveryReasonSaysSomethingAndNamesTheModel() {
        for reason in AutoModelReason.allCases {
            let sentence = reason.sentence(modelName: "Gemma 4 12B")
            XCTAssertFalse(sentence.isEmpty, "\(reason) has no sentence")
            XCTAssertTrue(sentence.contains("Gemma 4 12B"), "\(reason): \(sentence) doesn't name the model")
        }
    }
}
