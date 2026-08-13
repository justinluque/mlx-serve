import XCTest
@testable import MLXCore

/// Where a queued message is DELIVERED. The queue itself is pure and tested in
/// `MessageQueueTests`; these are the engine's two delivery points and the one
/// place delivery must NOT happen, none of which a pure test can reach — they
/// are positions in `ChatTurnEngine`'s control flow, so they are pinned by
/// reading it (the source-scan idiom used for the cold-load and route-order
/// invariants elsewhere).
final class MessageQueueDeliveryTests: XCTestCase {

    private func engineSource() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try String(contentsOf: root.appendingPathComponent(
            "Sources/MLXServe/Services/ChatTurnEngine.swift"), encoding: .utf8)
    }

    /// The whole point of the feature: a steer typed during a tool loop is read
    /// by the NEXT round, not after the loop finishes. The injection has to sit
    /// above the history build — history is rebuilt from the session every
    /// iteration, and a message appended after that build waits a full round.
    func testTheAgentLoopInjectsTheQueueBeforeItBuildsTheNextRequest() throws {
        let src = try engineSource()
        guard let loop = src.range(of: "private func runAgentLoop") else {
            return XCTFail("runAgentLoop not found — this guard has gone stale")
        }
        let body = String(src[loop.lowerBound...])
        guard let inject = body.range(of: "injectQueuedMessage(into: sessionId)") else {
            return XCTFail("the agent loop must inject the queue at its round boundary")
        }
        guard let history = body.range(of: "AgentEngine.buildAgentHistory") else {
            return XCTFail("buildAgentHistory not found — this guard has gone stale")
        }
        XCTAssertLessThan(inject.lowerBound, history.lowerBound,
                          "the queue is injected BEFORE the round's history is built, or the "
                          + "message it carries waits a whole extra round")
    }

    /// A user turn's budget resets with the user turn. An injected message IS a
    /// new user turn, so asking for an image mid-loop must not be refused
    /// against the budget the earlier message spent.
    func testInjectingAQueuedMessageMintsAFreshMediaTurn() throws {
        let src = try engineSource()
        guard let loop = src.range(of: "private func runAgentLoop") else {
            return XCTFail("runAgentLoop not found — this guard has gone stale")
        }
        let body = String(src[loop.lowerBound...])
        guard let inject = body.range(of: "if injectQueuedMessage(into: sessionId) {"),
              let close = body.range(of: "\n            }", range: inject.upperBound..<body.endIndex) else {
            return XCTFail("the injection branch not found — this guard has gone stale")
        }
        let branch = String(body[inject.upperBound..<close.lowerBound])
        XCTAssertTrue(branch.contains("mediaTurn = UUID()"),
                      "an injected message is a new user turn and gets a fresh media budget")
    }

    /// Stop must not fire a fresh generation off the back of a deliberate
    /// interruption. It is enforced structurally — `stop` drops the config the
    /// drain needs — so this pins that the drop is still there.
    func testStoppingATurnDropsTheContinuationSoNothingAutoSends() throws {
        let src = try engineSource()
        // The trailing newline discriminates the real method from the
        // `TurnRunning` extension's one-line default, which shares its prefix.
        guard let stop = src.range(of: "func stop(sessionId: UUID) {\n"),
              let end = src.range(of: "\n    }", range: stop.upperBound..<src.endIndex) else {
            return XCTFail("stop(sessionId:) not found — this guard has gone stale")
        }
        let body = String(src[stop.upperBound..<end.lowerBound])
        XCTAssertTrue(body.contains("turnContinuation.removeValue(forKey: sessionId)"),
                      "a stopped turn hands nothing on: the queue stays visible and unsent")
        XCTAssertFalse(body.contains("drainQueueAsNewTurn"),
                       "Stop must never start the queued message itself")
    }

    /// Both terminal paths deliver, and neither cancellation path does. A
    /// cancelled turn returns EARLY in both drivers; a drain reachable from
    /// those returns would restart the turn the user just stopped.
    func testEveryDrainSiteSitsOnASuccessfulTurnsExit() throws {
        let src = try engineSource()
        let drains = src.components(separatedBy: "drainQueueAsNewTurn(sessionId: sessionId)").count - 1
        XCTAssertEqual(drains, 2,
                       "exactly two delivery points — the agent turn's clean exit and plain "
                       + "chat's — plus the definition")
        // Each one is the last statement of its driver, immediately after
        // endTurn: a drain before endTurn would supersede itself (runTurn stops
        // the session's turn first) and never run.
        for driver in ["self.endTurn(sessionId: sessionId, token: token)\n            // Anything queued",
                       "endTurn(sessionId: sessionId, token: token)\n        // Plain chat"] {
            XCTAssertTrue(src.contains(driver),
                          "a drain runs AFTER endTurn releases the slot, or runTurn's own "
                          + "supersede would cancel the turn it is starting")
        }
    }

    /// A deleted chat's parked messages go with it — nothing else would ever
    /// clear them, and there is no conversation left to deliver them into.
    func testDeletingAChatTakesItsQueue() throws {
        let src = try engineSource()
        guard let sweep = src.range(of: "func stopIfOrphaned() {"),
              let end = src.range(of: "\n    }", range: sweep.upperBound..<src.endIndex) else {
            return XCTFail("stopIfOrphaned not found — this guard has gone stale")
        }
        let body = String(src[sweep.upperBound..<end.lowerBound])
        XCTAssertTrue(body.contains("queue.clear(sid)"),
                      "the orphaned-turn sweep clears the orphaned queue too")
    }
}
