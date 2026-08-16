import XCTest
@testable import MLXCore

/// The system prompt became user-facing in three ways: a custom file location,
/// an in-app editor, and an opt-in that extends it to PLAIN chat turns.
///
/// The third is the risky one. Plain chat has deliberately sent no system
/// message since the `formatNudge` incident (models read a synthesized system
/// message as the user's input), so the bar for this change is that the shipped
/// default is *provably* the old behaviour — not "probably fine".
final class SystemPromptExposureTests: XCTestCase {

    private var savedPath: String?

    override func setUp() {
        super.setUp()
        savedPath = UserDefaults.standard.string(forKey: AgentPrompt.promptPathDefaultsKey)
        UserDefaults.standard.removeObject(forKey: AgentPrompt.promptPathDefaultsKey)
    }

    override func tearDown() {
        if let savedPath {
            UserDefaults.standard.set(savedPath, forKey: AgentPrompt.promptPathDefaultsKey)
        } else {
            UserDefaults.standard.removeObject(forKey: AgentPrompt.promptPathDefaultsKey)
        }
        super.tearDown()
    }

    // MARK: - Plain chat opt-in

    func testPlainChatSendsNoBasePromptByDefault() {
        // The upgrade guarantee. If this ever returns non-nil with `enabled:
        // false`, every plain conversation in the app silently gains a system
        // message it never had.
        XCTAssertNil(ChatTurnEngine.plainChatBasePrompt(
            enabled: false, persona: "", prompt: "You are an autonomous agent."))
    }

    func testTheDefaultSettingIsOff() {
        XCTAssertFalse(ServerOptions().applyBasePromptToPlainChat)
    }

    func testEnabledPlainChatCarriesTheBasePrompt() {
        XCTAssertEqual(
            ChatTurnEngine.plainChatBasePrompt(
                enabled: true, persona: "", prompt: "  You are careful.  "),
            "You are careful.")
    }

    func testAPersonaBeatsTheBasePromptEvenWhenEnabled() {
        // Two identity claims in one turn is the "I'm poolside Malibu under an
        // Elon Musk persona" bug. The agent's prompt is the whole message.
        XCTAssertNil(ChatTurnEngine.plainChatBasePrompt(
            enabled: true, persona: "You are Ada.", prompt: "You are an autonomous agent."))
    }

    func testAWhitespaceOnlyPersonaDoesNotCountAsAPersona() {
        XCTAssertEqual(
            ChatTurnEngine.plainChatBasePrompt(
                enabled: true, persona: "   \n ", prompt: "You are careful."),
            "You are careful.")
    }

    func testAnEmptyPromptIsNilRatherThanAnEmptySystemMessage() {
        // `{"role":"system","content":""}` is not the same request as no system
        // message, and some templates render it as a stray marker.
        XCTAssertNil(ChatTurnEngine.plainChatBasePrompt(enabled: true, persona: "", prompt: ""))
        XCTAssertNil(ChatTurnEngine.plainChatBasePrompt(enabled: true, persona: "", prompt: "   \n"))
    }

    func testThePromptIsNotEvenReadWhenTheSettingIsOff() {
        // The autoclosure exists so a plain turn does no prompt-file I/O while
        // the feature is off — this runs on every message.
        var reads = 0
        func read() -> String { reads += 1; return "prompt" }
        _ = ChatTurnEngine.plainChatBasePrompt(enabled: false, persona: "", prompt: read())
        XCTAssertEqual(reads, 0, "the prompt file was read despite the setting being off")
        _ = ChatTurnEngine.plainChatBasePrompt(enabled: true, persona: "", prompt: read())
        XCTAssertEqual(reads, 1)
    }

    // MARK: - Custom prompt file location

    func testNoCustomPathMeansTheBuiltinLocation() {
        XCTAssertNil(AgentPrompt.customPromptPath)
        XCTAssertEqual(AgentPrompt.resolvedPromptPath, AgentPrompt.builtinPromptPath)
        XCTAssertFalse(AgentPrompt.customPromptIsMissing)
    }

    func testAWhitespaceOnlyCustomPathCountsAsUnset() {
        UserDefaults.standard.set("   ", forKey: AgentPrompt.promptPathDefaultsKey)
        XCTAssertNil(AgentPrompt.customPromptPath)
        XCTAssertEqual(AgentPrompt.resolvedPromptPath, AgentPrompt.builtinPromptPath)
    }

    func testACustomPathIsTildeExpanded() {
        UserDefaults.standard.set("~/prompts/mine.md", forKey: AgentPrompt.promptPathDefaultsKey)
        let expected = NSString(string: "~/prompts/mine.md").expandingTildeInPath
        XCTAssertEqual(AgentPrompt.customPromptPath, expected)
        XCTAssertEqual(AgentPrompt.resolvedPromptPath, expected)
    }

    func testAMissingCustomFileIsReportedRatherThanSilentlyFallingBack() {
        // The `huggingFaceRoot` rule: the point of the setting is that the file
        // lives elsewhere, so a quiet fall-through is how someone edits a file
        // for an hour that nothing is reading. The path stays the configured
        // one and the misconfiguration is nameable.
        let missing = NSTemporaryDirectory() + "definitely-not-here-\(UUID().uuidString).md"
        UserDefaults.standard.set(missing, forKey: AgentPrompt.promptPathDefaultsKey)
        XCTAssertEqual(AgentPrompt.resolvedPromptPath, missing)
        XCTAssertTrue(AgentPrompt.customPromptIsMissing)
        // …and the agent still gets a usable prompt rather than none at all.
        XCTAssertEqual(AgentPrompt.systemPrompt, AgentPrompt.defaultPromptFile)
    }

    func testAMissingCustomFileIsNotAutoCreated() {
        // Creating a file at a user-typed path turns a typo into a stray prompt
        // file that looks like it is working.
        let missing = NSTemporaryDirectory() + "not-created-\(UUID().uuidString).md"
        UserDefaults.standard.set(missing, forKey: AgentPrompt.promptPathDefaultsKey)
        _ = AgentPrompt.systemPrompt
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing))
    }

    // MARK: - In-app editor round trip

    func testTheEditorReadsRawBytesNotTheResolvedPrompt() throws {
        // An empty file must read as EMPTY in the editor. Pouring the built-in
        // default in and saving it back would turn "I have no prompt" into "I
        // have a copy of the default I now maintain by hand".
        let path = NSTemporaryDirectory() + "prompt-\(UUID().uuidString).md"
        try "".write(toFile: path, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: path) }
        UserDefaults.standard.set(path, forKey: AgentPrompt.promptPathDefaultsKey)

        XCTAssertEqual(AgentPrompt.promptFileContents(), "")
        // …while the SERVED prompt still falls back to the default.
        XCTAssertEqual(AgentPrompt.systemPrompt, AgentPrompt.defaultPromptFile)
    }

    func testSaveThenReadRoundTripsThroughTheCustomFile() throws {
        let path = NSTemporaryDirectory() + "prompt-\(UUID().uuidString).md"
        defer { try? FileManager.default.removeItem(atPath: path) }
        UserDefaults.standard.set(path, forKey: AgentPrompt.promptPathDefaultsKey)

        let mine = "Check before you answer. Never invent an API."
        XCTAssertTrue(AgentPrompt.savePromptFile(mine))
        XCTAssertEqual(AgentPrompt.promptFileContents(), mine)
        XCTAssertEqual(AgentPrompt.systemPrompt, mine)
    }

    func testSavingCreatesTheParentDirectory() throws {
        let dir = NSTemporaryDirectory() + "prompts-\(UUID().uuidString)"
        let path = (dir as NSString).appendingPathComponent("mine.md")
        defer { try? FileManager.default.removeItem(atPath: dir) }
        UserDefaults.standard.set(path, forKey: AgentPrompt.promptPathDefaultsKey)

        XCTAssertTrue(AgentPrompt.savePromptFile("hello"))
        XCTAssertEqual(AgentPrompt.promptFileContents(), "hello")
    }

    func testTheLegacyStubMigrationNeverWritesToAUserOwnedFile() throws {
        // The in-place migration exists to keep OUR file and the editor in
        // sync. Rewriting a file the user pointed us at is not ours to do.
        let path = NSTemporaryDirectory() + "prompt-\(UUID().uuidString).md"
        let stub = "These are appended to the base system prompt"
        try stub.write(toFile: path, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: path) }
        UserDefaults.standard.set(path, forKey: AgentPrompt.promptPathDefaultsKey)

        XCTAssertEqual(AgentPrompt.systemPrompt, AgentPrompt.defaultPromptFile)
        XCTAssertEqual(try String(contentsOfFile: path, encoding: .utf8), stub,
                       "the user's own file was rewritten by the stub migration")
    }
}
