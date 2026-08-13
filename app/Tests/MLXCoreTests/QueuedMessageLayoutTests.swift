import XCTest
@testable import MLXCore

/// Where a queued message is DRAWN.
///
/// A parked message is the user's next turn, so it belongs at the bottom of the
/// conversation — not in a strip above the composer, which read as composer
/// state (like the attachment row it sat beside) rather than as something about
/// to be said. Drawing it as the last row of the transcript also settles the
/// ordering for free: a message delivered mid-turn is appended to
/// `session.messages`, which renders ABOVE the queued rows, so the reply to it
/// and anything queued afterwards land underneath — no separate ordering rule
/// to keep in sync with the engine's delivery points.
///
/// Source audit because there is no seam: a strip re-added above the composer
/// compiles and renders exactly as well as the row in the transcript does.
final class QueuedMessageLayoutTests: XCTestCase {

    private func chatSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views/ChatView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testQueuedMessagesRenderInsideTheTranscript() throws {
        let chat = try chatSource()
        let rows = try XCTUnwrap(chat.range(of: "ChatRowBuilder.rows("),
                                 "the transcript still builds its rows here")
        let queued = try XCTUnwrap(chat.range(of: "QueuedMessageBubble("),
                                   "queued messages must be drawn as transcript rows")
        // The transcript ScrollView's content ends before the modifiers chained
        // onto it, of which this is the first.
        let scrollEnd = try XCTUnwrap(chat.range(of: ".scrollEdgeEffectStyle(.soft, for: .top)"),
                                      "the transcript scroll view still carries the edge effect")
        XCTAssertTrue(rows.upperBound < queued.lowerBound,
                      "queued rows go AFTER the conversation's own messages")
        XCTAssertTrue(queued.upperBound < scrollEnd.lowerBound,
                      "queued rows must be INSIDE the transcript scroll view, not above the composer")
    }

    func testQueuedRowsAreTheLastThingInTheTranscript() throws {
        let chat = try chatSource()
        let media = try XCTUnwrap(chat.range(of: "MediaProgressCard(progress: progress)"),
                                  "the live media card still renders in the transcript")
        let queued = try XCTUnwrap(chat.range(of: "QueuedMessageBubble("),
                                   "queued messages must be drawn as transcript rows")
        // Everything belonging to the RUNNING turn draws above what is still
        // waiting to be sent — that is what makes "the reply lands under your
        // queued message" true by layout rather than by an ordering rule.
        XCTAssertTrue(media.upperBound < queued.lowerBound,
                      "the running turn's media card belongs above the queued rows")
    }

    func testNothingQueuedIsDrawnInTheComposerStack() throws {
        let chat = try chatSource()
        XCTAssertFalse(chat.contains("QueuedMessagesStrip"),
                       "the above-the-composer strip is retired — two places to show a queue is two designs")
    }

    /// A chat holding nothing but a parked message still renders its transcript:
    /// the empty-state branch draws no rows at all, so the queue would be
    /// invisible while the Send button stayed enabled by it (reachable after a
    /// stopped turn whose messages were deleted).
    func testAQueueKeepsTheChatOutOfTheEmptyState() throws {
        let chat = try chatSource()
        let start = try XCTUnwrap(chat.range(of: "private var isEmptyConversation: Bool {"))
        let rest = chat[start.upperBound...]
        let end = try XCTUnwrap(rest.range(of: "}"))
        XCTAssertTrue(rest[..<end.lowerBound].contains("queuedMessages.isEmpty"),
                      "a non-empty queue means there is something to draw")
    }

    // MARK: - What a row says

    func testAQueuedRowNamesAttachmentsWhenThereIsNoText() {
        let img = QueuedMessage(text: "", images: [ChatImage(data: Data([0x1]))])
        XCTAssertEqual(QueuedMessageBubble.preview(img), "Image")
        let two = QueuedMessage(text: "", images: [
            ChatImage(data: Data([0x1])),
            ChatImage(data: Data([0x2])),
        ])
        XCTAssertEqual(QueuedMessageBubble.preview(two), "2 images")
        let clip = QueuedMessage(text: "", audio: [
            ChatAudio(name: "take.wav", pcm: Data(repeating: 0, count: 64)),
        ])
        XCTAssertEqual(QueuedMessageBubble.preview(clip), "Audio clip")
    }

    func testAQueuedRowShowsItsTextVerbatim() {
        // In the transcript there is room for the whole message — the strip's
        // one-line elision was a property of sitting in the composer row.
        let m = QueuedMessage(text: "first line\nsecond line")
        XCTAssertEqual(QueuedMessageBubble.preview(m), "first line\nsecond line")
    }
}
