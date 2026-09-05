import XCTest
@testable import MLXCore

/// Ideogram 4's app-side facts, each pinned to a CONCRETE server-side one.
/// The pane offers a control only where the backend honours it — a setting the
/// server ignores is worse than a missing one, because the user pays attention
/// to it and gets nothing.
final class Ideogram4PresetTests: XCTestCase {

    private var presets: [ImageModelPreset] {
        ImageModelPreset.all.filter { $0.variant == .ideogram4 }
    }

    /// The catalog ships only packs that RENDER. A `mixed_2_8` build was
    /// published here and withdrawn: 2-bit affine on the DiT bulk renders a
    /// woven grid texture at every prompt, seed and resolution, while
    /// `mixed_3_8` renders the same prompts correctly — so 3 bits is the floor
    /// and the converter refuses to mint another one (`MIN_BULK_BITS` in
    /// `tests/convert_ideogram4.py`). A quantization the catalog offers is a
    /// promise that it works; a smaller download is not worth breaking it.
    /// Two sizes ship: `mixed_3_8` for the 24 GB / 6-GB-not-guaranteed floor,
    /// `mixed_4_8` for a Mac with more headroom to spend on quality.
    func testTheCatalogShipsOnlyThePacksThatRender() {
        XCTAssertEqual(presets.map(\.repo), [
            "justintime47/Ideogram-4-MLX-Serve-mixed_3_8",
            "justintime47/Ideogram-4-MLX-Serve-mixed_4_8",
        ])
        for p in ImageModelPreset.all {
            XCTAssertFalse(p.repo.hasSuffix("_2_8"), "\(p.id) is a withdrawn 2-bit pack")
        }
    }

    func testTheCatalogIsOrderedByDownloadSize() {
        let sizes = ImageModelPreset.all.map { $0.approxDownloadGB }
        XCTAssertEqual(sizes, sizes.sorted(), "the catalog is ordered cheapest → heaviest")
    }

    /// `configName` is what the server matches on. Anything else and the pack
    /// downloads, appears in the picker, and then fails to route.
    func testEveryPackDeclaresTheServersModelType() {
        for p in presets {
            XCTAssertEqual(p.configName, "ideogram4", "\(p.id)")
        }
    }

    /// The DiT patchifies ×2 over a ×8 VAE, so every side must be a multiple
    /// of 16 — mirrors `ideogram4.clampDim` and its 256–2048 window.
    func testResolutionGridMirrorsTheServersClamp() {
        for p in presets {
            XCTAssertEqual(p.resolutionGrid.alignment, 16, "\(p.id)")
            XCTAssertEqual(p.resolutionGrid.minDim, 256, "\(p.id)")
            XCTAssertEqual(p.resolutionGrid.maxDim, 2048, "\(p.id)")
            for r in p.resolutions {
                XCTAssertEqual(r.width % 16, 0, "\(p.id): \(r.width)×\(r.height)")
                XCTAssertEqual(r.height % 16, 0, "\(p.id): \(r.width)×\(r.height)")
                XCTAssertTrue((256...2048).contains(r.width), "\(p.id): \(r.width)")
                XCTAssertTrue((256...2048).contains(r.height), "\(p.id): \(r.height)")
            }
        }
    }

    /// The step count is what SELECTS the sampler preset server-side
    /// (`ideogram4.SamplerPreset.forSteps`), and each preset carries its own
    /// mu/std. A tier that lands near 20 instead of on it silently runs the
    /// 20-step schedule's noise curve at a different step count.
    func testQualityTiersLandOnTheReferencesOwnStepCounts() {
        let sampler: Set<Int> = [12, 20, 48]
        for p in presets {
            for tier in QualityPreset.allCases {
                XCTAssertTrue(sampler.contains(p.settings(tier).steps),
                              "\(p.id) \(tier): \(p.settings(tier).steps) is not a sampler preset")
            }
            XCTAssertEqual(p.settings(.fast).steps, 12)
            XCTAssertEqual(p.settings(.good).steps, 20)
            XCTAssertEqual(p.settings(.quality).steps, 48)
            // Steps are a real knob here: the schedule is not distilled shut.
            XCTAssertFalse(p.stepsAreFixed, "\(p.id)")
        }
    }

    /// Capabilities, each against the matching arm in `gen.ImageEngine`.
    func testCapabilityFlagsMatchTheBackend() {
        for p in presets {
            // The Flux2 autoencoder ships BOTH halves and `convert_ideogram4.py`
            // carries them both into the pack, so `mode:"variation"` is a real
            // path (#9eed64d) — not the decoder-only pack this test first
            // assumed.
            XCTAssertTrue(p.supportsImg2Img, "\(p.id)")
            // No edit training, and no in-context reference conditioning.
            XCTAssertFalse(p.supportsReferenceEdit, "\(p.id)")
            // Runtime LoRA attaches to the conditional transformer.
            XCTAssertTrue(p.supportsLoRA, "\(p.id)")
            // 13 taps, but the rebalance UI slices tap-major features and
            // Ideogram's are tap-inner — gen.zig reports 0 until that is wired.
            XCTAssertEqual(p.condWeightCount, 0, "\(p.id)")
            // The one control that is Ideogram-only.
            XCTAssertTrue(p.supportsMagicPrompt, "\(p.id)")
        }
    }

    /// And nothing else claims it: a plain sentence is out of distribution for
    /// Ideogram specifically, not for image models in general.
    func testMagicPromptIsOfferedNowhereElse() {
        for p in ImageModelPreset.all where p.variant != .ideogram4 {
            XCTAssertFalse(p.supportsMagicPrompt, "\(p.id)")
        }
    }

    /// Discovery has to classify the pack from either shape: our converted
    /// pack writes a root `model_type`, an unconverted upstream repo carries
    /// only `model_index.json`. Mirrors `model_discovery.peekIdeogram4Index`.
    func testBothCheckpointShapesClassifyAsImage() {
        XCTAssertEqual(MediaModality(modelType: "ideogram4"), .image)
        XCTAssertNotNil(MediaModality(modelType: "ideogram4"))
    }
}

/// The request body: `magic_prompt` travels only where the server acts on it.
@MainActor
final class Ideogram4RequestBodyTests: XCTestCase {

    private func request(_ preset: ImageModelPreset, magic: Bool = true, rewriter: String = "") -> ImageGenRequest {
        var r = ImageGenRequest(model: preset, prompt: "a red barn", width: 1024, height: 1024, steps: 20)
        r.magicPrompt = magic
        r.magicPromptModel = rewriter
        return r
    }

    func testMagicPromptIsSentExplicitlyForIdeogram() {
        let p = ImageModelPreset.all.first { $0.variant == .ideogram4 }!
        let on = ImageGenService.requestJson(for: request(p), modelName: "m", seed: 1)
        XCTAssertEqual(on["magic_prompt"] as? Bool, true)
        // OFF must be sent too: the server's own default is `auto`, which would
        // rewrite a caption the user deliberately hand-wrote.
        let off = ImageGenService.requestJson(for: request(p, magic: false), modelName: "m", seed: 1)
        XCTAssertEqual(off["magic_prompt"] as? Bool, false)
    }

    func testTheRewriterModelRidesAlongOnlyWhenNamedAndEnabled() {
        let p = ImageModelPreset.all.first { $0.variant == .ideogram4 }!
        let named = ImageGenService.requestJson(for: request(p, rewriter: "org/repo"), modelName: "m", seed: 1)
        XCTAssertEqual(named["magic_prompt_model"] as? String, "org/repo")
        // Empty = the server's default text model; sending "" would resolve to
        // the default anyway, but says something the user did not ask for.
        let unnamed = ImageGenService.requestJson(for: request(p), modelName: "m", seed: 1)
        XCTAssertNil(unnamed["magic_prompt_model"])
        // Naming a rewriter while the toggle is off is not a request to rewrite.
        let offNamed = ImageGenService.requestJson(for: request(p, magic: false, rewriter: "org/repo"), modelName: "m", seed: 1)
        XCTAssertNil(offNamed["magic_prompt_model"])
    }

    func testEveryOtherBackendSeesNoMagicPromptFieldAtAll() {
        for p in ImageModelPreset.all where p.variant != .ideogram4 {
            let json = ImageGenService.requestJson(for: request(p), modelName: "m", seed: 1)
            XCTAssertNil(json["magic_prompt"], "\(p.id)")
            XCTAssertNil(json["magic_prompt_model"], "\(p.id)")
        }
    }
}

/// The rewriter field takes a model ID, but everything the user can SEE is a
/// display label. A label pasted (or persisted from the old free-text field)
/// into that slot resolved to nothing server-side and the whole rewrite fell
/// back to the raw prompt — which for Ideogram 4 means conditioning on
/// out-of-distribution text, with only a log line to say so.
@MainActor
final class MagicPromptRewriterPickTests: XCTestCase {

    private func chat(_ name: String, lan: String? = nil) -> ModelInfo {
        ModelInfo(name: name, quantBits: 4, layers: 0, hiddenSize: 0, vocabSize: 0,
                  contextLength: 0, modelMaxTokens: 0, capabilities: ["chat"], lanPeer: lan)
    }
    private func image(_ name: String) -> ModelInfo {
        ModelInfo(name: name, quantBits: 4, layers: 0, hiddenSize: 0, vocabSize: 0,
                  contextLength: 0, modelMaxTokens: 0, capabilities: ["image"])
    }

    /// Only local chat models can write a caption: a generator has no
    /// tokenizer, and a LAN entry is the peer's to load.
    func testOnlyLocalChatModelsAreOffered() {
        let models = [chat("a"), image("b"), chat("c@peer", lan: "peer")]
        XCTAssertEqual(MagicPromptRewriter.choices(models).map(\.name), ["a"])
    }

    /// The empty string is the documented "server default" and must survive
    /// normalization untouched.
    func testEmptyStaysTheServerDefault() {
        XCTAssertEqual(MagicPromptRewriter.normalize("  ", in: [chat("a")]), "")
    }

    /// A saved value that is already an id is returned verbatim, even when a
    /// looser match would also fit.
    func testAnExactIdWins() {
        let models = [chat("Gemma-4-E4B"), chat("Gemma-4-E4B-Q8_K_P")]
        XCTAssertEqual(MagicPromptRewriter.normalize("Gemma-4-E4B", in: models), "Gemma-4-E4B")
    }

    /// The two forms the picker/tray actually render: `org/repo · QUANT` and
    /// the bare repo name. Both map back onto the served id.
    func testADisplayLabelMapsBackOntoItsId() {
        let models = [chat("Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P"), chat("Qwen3.5-4B")]
        XCTAssertEqual(
            MagicPromptRewriter.normalize("HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive · Q8_K_P", in: models),
            "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P")
        XCTAssertEqual(
            MagicPromptRewriter.normalize("Gemma-4-E4B-Uncensored-HauhauCS-Aggressive", in: models),
            "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P")
    }

    /// Ambiguity is not repaired: two quants of one repo are two different
    /// models, and picking one for the user is a silent model swap.
    func testAnAmbiguousLabelIsLeftAlone() {
        let models = [chat("repo-Q4_K_M"), chat("repo-Q8_K_P")]
        XCTAssertEqual(MagicPromptRewriter.normalize("repo", in: models), "repo")
    }

    /// A model that is simply gone (deleted, or the server is down and the
    /// list is empty) keeps its saved name so the UI can say "unavailable"
    /// instead of quietly switching to the default.
    func testAnUnknownNameIsPreserved() {
        XCTAssertEqual(MagicPromptRewriter.normalize("gone", in: []), "gone")
        XCTAssertEqual(MagicPromptRewriter.normalize("gone", in: [chat("a")]), "gone")
    }
}

/// The magic-prompt controls call `persist()` on every change, so they are
/// meant to be sticky — but neither one was in `ImageGenSettings`, so the
/// rewriter choice was silently forgotten at every relaunch.
@MainActor
final class MagicPromptStickinessTests: XCTestCase {
    func testTheMagicPromptControlsSurviveASaveLoadRoundTrip() throws {
        var s = ImageGenSettings()
        XCTAssertTrue(s.magicPrompt, "on by default — Ideogram wants a caption")
        XCTAssertEqual(s.magicPromptModel, "", "no rewriter named = the server's default")
        s.magicPrompt = false
        s.magicPromptModel = "org/repo"
        let back = try JSONDecoder().decode(ImageGenSettings.self,
                                            from: try JSONEncoder().encode(s))
        XCTAssertEqual(back.magicPrompt, false)
        XCTAssertEqual(back.magicPromptModel, "org/repo")
    }

    /// Settings written by a build that had neither field still decode.
    func testOlderSettingsDecodeWithTheDefaults() throws {
        let legacy = #"{"modelId":"x","quality":"Good","resolutionId":"r","steps":8,"seed":-1,"keepResident":false,"strength":0.6,"editMode":true,"condGain":1.0,"condWeightsText":"","loras":[],"customWidth":1024,"customHeight":1024}"#
        let s = try JSONDecoder().decode(ImageGenSettings.self, from: Data(legacy.utf8))
        XCTAssertTrue(s.magicPrompt)
        XCTAssertEqual(s.magicPromptModel, "")
    }
}
