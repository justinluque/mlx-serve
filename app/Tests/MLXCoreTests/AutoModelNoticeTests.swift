import XCTest
@testable import MLXCore

/// The transcript row that says which model was picked and why, plus the gate
/// that decides whether Send is live at all.
///
/// The point of the feature is that the composer stops refusing input, so the
/// gate is where it can silently come back: every "disabled" below is a case
/// where sending would have nowhere to go.
final class AutoModelNoticeTests: XCTestCase {

    // MARK: - Copy

    func testEveryStateSaysWhatHappenedAndNamesTheModel() {
        let kinds: [AutoModelNotice.Kind] = [.loading, .loaded, .needsDownload, .failed]
        for kind in kinds {
            let notice = AutoModelNotice(kind: kind, modelName: "Gemma 4 12B", reason: .vision)
            XCTAssertTrue(notice.headline.contains("Gemma 4 12B"),
                          "\(kind) headline doesn't name the model: \(notice.headline)")
            XCTAssertFalse(notice.detail.isEmpty, "\(kind) has no detail line")
        }
    }

    func testAFailureShowsWhatTheServerSaid() {
        let notice = AutoModelNotice(kind: .failed, modelName: "Gemma 4 12B", reason: .ramFit,
                                     failureMessage: "not enough memory")
        XCTAssertTrue(notice.detail.contains("not enough memory"))
    }

    func testAFailureWithNothingToQuoteStillExplainsItself() {
        // Inventing a diagnosis sends people to change settings that were never
        // the problem — but silence is worse than a generic sentence here.
        let notice = AutoModelNotice(kind: .failed, modelName: "Gemma 4 12B", reason: .ramFit)
        XCTAssertFalse(notice.detail.isEmpty)
    }

    func testTheNoticeSurvivesASaveAndReload() throws {
        // It rides `ChatMessage`, which is persisted to chat-history.json.
        let notice = AutoModelNotice(kind: .needsDownload, modelName: "Gemma 4 E4B",
                                     reason: .vision, repoId: "mlx-community/gemma-4-e4b-it-4bit",
                                     queuedPreview: "what's in this picture?")
        let data = try JSONEncoder().encode(notice)
        XCTAssertEqual(try JSONDecoder().decode(AutoModelNotice.self, from: data), notice)
    }

    func testAMessageWithoutANoticeDecodesFromAnOlderTranscript() throws {
        // Sessions written before this feature carry no key at all.
        let json = #"{"id":"\#(UUID().uuidString)","role":"user","content":"hi","isStreaming":false,"timestamp":0}"#
        let msg = try JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
        XCTAssertNil(msg.modelNotice)
    }

    // MARK: - Send gate

    func testAnEmptyComposerStaysDisabled() {
        XCTAssertEqual(ChatSendGate.resolve(status: .running, hasContent: false,
                                            hasAutoPick: true, isAwaitingAutoModel: false),
                       .disabled)
    }

    func testARunningServerSendsNormally() {
        XCTAssertEqual(ChatSendGate.resolve(status: .running, hasContent: true,
                                            hasAutoPick: true, isAwaitingAutoModel: false),
                       .send)
    }

    func testAStoppedServerWithModelsOnDiskPicksOneInsteadOfRefusing() {
        // This is the whole feature: the composer used to be dead here.
        for status in [ServerStatus.stopped, .error("port in use")] {
            XCTAssertEqual(ChatSendGate.resolve(status: status, hasContent: true,
                                                hasAutoPick: true, isAwaitingAutoModel: false),
                           .autoPick, "\(status) should auto-pick")
        }
    }

    func testWithNothingToPickTheButtonIsStillDisabled() {
        // The blocking gate sheet is already on screen offering the download;
        // a live Send with nowhere to send is the dead-control class.
        XCTAssertEqual(ChatSendGate.resolve(status: .stopped, hasContent: true,
                                            hasAutoPick: false, isAwaitingAutoModel: false),
                       .disabled)
    }

    func testAMessageAlreadyWaitingOnAModelDoesNotQueueASecond() {
        XCTAssertEqual(ChatSendGate.resolve(status: .stopped, hasContent: true,
                                            hasAutoPick: true, isAwaitingAutoModel: true),
                       .disabled)
    }

    // MARK: - Wiring
    //
    // The gate is only worth anything if the composer actually asks it. These
    // scan the source because the alternative is a UI test for a `.disabled`
    // modifier — and the failure they catch (someone re-adding a direct
    // `server.status != .running` check to Send) is exactly the refusal this
    // feature exists to remove.

    private func chatViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views/ChatView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testSendAsksTheGateRatherThanTheServerStatus() throws {
        let source = try chatViewSource()
        XCTAssertTrue(source.contains("ChatSendGate.resolve("),
                      "the composer must decide Send through ChatSendGate")
        XCTAssertTrue(source.contains(".disabled(composerState == .idle && sendGate == .disabled)"),
                      "Send's disabled binding must read the gate, not the server status")
    }

    func testTheColdStartPathRunsThroughThePicker() throws {
        let source = try chatViewSource()
        XCTAssertTrue(source.contains("AutoModelPicker.pick("),
                      "the auto-pick branch must ask AutoModelPicker")
        XCTAssertTrue(source.contains("useModelAndAwaitReady"),
                      "a picked model must be loaded through the one load-and-wait call")
    }

    func testTheNoticeRowIsExcludedFromTheHistorySentBack() throws {
        // An assistant turn describing our own machinery is how a model starts
        // explaining it back to you (the error-card rule).
        let source = try chatViewSource()
        let marker = "msg.modelNotice = notice"
        let idx = try XCTUnwrap(source.range(of: marker))
        let after = source[idx.upperBound...].prefix(200)
        XCTAssertTrue(after.contains("msg.failedRetry = true"),
                      "the auto-model row must be marked failedRetry beside the notice")
    }

    func testAServerAlreadyComingUpIsWaitedFor() {
        // `.starting` means a load is in flight — queueing a second pick on top
        // of it is how two models end up loading at once.
        XCTAssertEqual(ChatSendGate.resolve(status: .starting, hasContent: true,
                                            hasAutoPick: true, isAwaitingAutoModel: false),
                       .disabled)
    }
}
