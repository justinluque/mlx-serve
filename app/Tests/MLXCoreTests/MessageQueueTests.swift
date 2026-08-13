import XCTest
@testable import MLXCore

/// Message queueing: typing while a turn is in flight parks the message instead
/// of being swallowed, and it is delivered at the next boundary — for an agent
/// turn that is the next TOOL ROUND, not the end of a 150-iteration loop.
///
/// Everything decided here is pure: what a Return does, whether a submit sends
/// or queues, and what the drained message looks like. The engine wiring reads
/// these answers; it never re-derives them.
final class MessageQueueTests: XCTestCase {

    private func msg(_ text: String) -> QueuedMessage { QueuedMessage(text: text) }

    // MARK: - Submit action (send vs queue vs ignore)

    func testSubmitSendsWhenIdle() {
        XCTAssertEqual(ComposerSubmitAction.resolve(generating: false, serverRunning: true, hasContent: true),
                       .send)
    }

    func testSubmitQueuesWhileThisChatIsGenerating() {
        // The whole feature: a Return mid-turn parks the message. It used to be
        // swallowed outright (ComposerReturnAction.ignore).
        XCTAssertEqual(ComposerSubmitAction.resolve(generating: true, serverRunning: true, hasContent: true),
                       .queue)
    }

    func testEmptyComposerNeverQueuesOrSends() {
        XCTAssertEqual(ComposerSubmitAction.resolve(generating: true, serverRunning: true, hasContent: false),
                       .ignore)
        XCTAssertEqual(ComposerSubmitAction.resolve(generating: false, serverRunning: true, hasContent: false),
                       .ignore)
    }

    func testNothingIsQueuedWhileTheServerIsDown() {
        // A queued message is a promise to deliver it. With the server down
        // there is no turn to deliver it at the end of, so the promise would
        // be a lie — Send is already disabled in that state.
        XCTAssertEqual(ComposerSubmitAction.resolve(generating: false, serverRunning: false, hasContent: true),
                       .ignore)
        XCTAssertEqual(ComposerSubmitAction.resolve(generating: true, serverRunning: false, hasContent: true),
                       .ignore)
    }

    // MARK: - Return key

    func testShiftReturnIsAlwaysANewline() {
        XCTAssertEqual(ComposerKey.onReturn(shift: true), .newline)
    }

    func testBareReturnAlwaysSubmits() {
        // The keypress no longer decides send-vs-swallow: it submits, and
        // ComposerSubmitAction decides what a submit MEANS. Deciding it in two
        // places is how the field would swallow a Return the composer could
        // have queued.
        XCTAssertEqual(ComposerKey.onReturn(shift: false), .submit)
    }

    // MARK: - Queue mechanics

    func testEnqueuePreservesOrder() {
        var q = MessageQueue()
        let s = UUID()
        XCTAssertTrue(q.enqueue(msg("first"), for: s))
        XCTAssertTrue(q.enqueue(msg("second"), for: s))
        XCTAssertEqual(q.messages(for: s).map(\.text), ["first", "second"])
    }

    func testEnqueueTrimsAndRefusesAnEmptyMessage() {
        var q = MessageQueue()
        let s = UUID()
        XCTAssertFalse(q.enqueue(msg("   \n "), for: s), "nothing to queue")
        XCTAssertTrue(q.messages(for: s).isEmpty)
        XCTAssertTrue(q.enqueue(msg("  hi  "), for: s))
        XCTAssertEqual(q.messages(for: s).map(\.text), ["hi"])
    }

    func testAnAttachmentAloneIsQueueable() {
        var q = MessageQueue()
        let s = UUID()
        let img = ChatImage(data: Data([0xFF, 0xD8]))
        XCTAssertTrue(q.enqueue(QueuedMessage(text: "", images: [img]), for: s),
                      "an image with no caption is still a message")
        XCTAssertEqual(q.messages(for: s).count, 1)
    }

    func testQueuesAreScopedPerSession() {
        var q = MessageQueue()
        let a = UUID(), b = UUID()
        _ = q.enqueue(msg("for a"), for: a)
        XCTAssertTrue(q.messages(for: b).isEmpty,
                      "one chat's queue must not appear in another's composer")
        XCTAssertFalse(q.isEmpty(for: a))
        XCTAssertTrue(q.isEmpty(for: b))
    }

    func testRemoveTakesOneEntryAndClearTakesTheSession() {
        var q = MessageQueue()
        let s = UUID()
        let one = msg("one"), two = msg("two")
        _ = q.enqueue(one, for: s)
        _ = q.enqueue(two, for: s)
        q.remove(one.id, from: s)
        XCTAssertEqual(q.messages(for: s).map(\.text), ["two"])
        q.clear(s)
        XCTAssertTrue(q.isEmpty(for: s))
    }

    // MARK: - Draining

    func testDrainCombinesEveryPendingMessageIntoOne() {
        var q = MessageQueue()
        let s = UUID()
        _ = q.enqueue(msg("stop using tabs"), for: s)
        _ = q.enqueue(msg("and check the tests"), for: s)
        let drained = q.drain(s)
        XCTAssertEqual(drained?.text, "stop using tabs\n\nand check the tests")
        XCTAssertTrue(q.isEmpty(for: s), "draining empties the queue")
    }

    func testDrainCarriesAttachmentsAcrossEveryQueuedMessage() {
        var q = MessageQueue()
        let s = UUID()
        let a = ChatImage(data: Data([0x01])), b = ChatImage(data: Data([0x02]))
        let clip = ChatAudio(name: "note.wav", pcm: Data([0, 0, 0, 0]))
        _ = q.enqueue(QueuedMessage(text: "look", images: [a]), for: s)
        _ = q.enqueue(QueuedMessage(text: "and this", images: [b], audio: [clip]), for: s)
        let drained = q.drain(s)
        XCTAssertEqual(drained?.images?.count, 2)
        XCTAssertEqual(drained?.audio?.count, 1)
    }

    func testDrainKeepsAbsentAttachmentsNil() {
        // nil and [] are different to every consumer downstream (`runTurn`
        // guards on `images != nil`), so an all-text drain must stay nil.
        var q = MessageQueue()
        let s = UUID()
        _ = q.enqueue(msg("text only"), for: s)
        let drained = q.drain(s)
        XCTAssertNil(drained?.images)
        XCTAssertNil(drained?.audio)
    }

    func testDrainingAnEmptyQueueYieldsNothing() {
        var q = MessageQueue()
        XCTAssertNil(q.drain(UUID()))
    }

    func testDrainOfACaptionlessAttachmentKeepsEmptyText() {
        var q = MessageQueue()
        let s = UUID()
        _ = q.enqueue(QueuedMessage(text: "", images: [ChatImage(data: Data([0x01]))]), for: s)
        let drained = q.drain(s)
        XCTAssertEqual(drained?.text, "")
        XCTAssertEqual(drained?.images?.count, 1)
    }

    func testDrainSkipsBlankTextWhenJoining() {
        // A captionless image queued between two typed messages must not open
        // a hole of blank lines in the delivered text.
        var q = MessageQueue()
        let s = UUID()
        _ = q.enqueue(msg("one"), for: s)
        _ = q.enqueue(QueuedMessage(text: "", images: [ChatImage(data: Data([0x01]))]), for: s)
        _ = q.enqueue(msg("two"), for: s)
        XCTAssertEqual(q.drain(s)?.text, "one\n\ntwo")
    }

    func testDrainingOneSessionLeavesTheOtherPending() {
        var q = MessageQueue()
        let a = UUID(), b = UUID()
        _ = q.enqueue(msg("a"), for: a)
        _ = q.enqueue(msg("b"), for: b)
        _ = q.drain(a)
        XCTAssertEqual(q.messages(for: b).map(\.text), ["b"])
    }
}
