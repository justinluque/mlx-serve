import Foundation

/// A user message typed while a turn was already running in that chat.
///
/// It is NOT a `ChatMessage`: nothing about it is in the transcript yet, it has
/// no place in the history sent to the model, and it is deletable individually
/// right up until it is delivered. It becomes a `ChatMessage` at exactly one
/// moment — when the engine drains it into the conversation.
struct QueuedMessage: Identifiable, Equatable {
    let id: UUID
    var text: String
    var images: [ChatImage]?
    var audio: [ChatAudio]?

    init(id: UUID = UUID(), text: String, images: [ChatImage]? = nil, audio: [ChatAudio]? = nil) {
        self.id = id
        self.text = text
        self.images = images
        self.audio = audio
    }

    /// Something to deliver — an image with no caption is still a message.
    var hasContent: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !(images ?? []).isEmpty
            || !(audio ?? []).isEmpty
    }
}

/// Per-session queue of messages typed during a turn. Pure — the engine owns an
/// instance and publishes a mirror of it; every decision about ORDER, combining
/// and emptiness is made here so the two delivery points (the agent loop's round
/// boundary and the end of a turn) cannot disagree about what gets delivered.
struct MessageQueue {
    private var pending: [UUID: [QueuedMessage]] = [:]

    init() {}

    func messages(for session: UUID) -> [QueuedMessage] { pending[session] ?? [] }
    func isEmpty(for session: UUID) -> Bool { messages(for: session).isEmpty }

    /// The whole queue, for the engine's published mirror. Sessions with nothing
    /// pending are ABSENT rather than empty — every mutating path clears its key
    /// down to nil, so an empty array can never be published as "still queued".
    var snapshot: [UUID: [QueuedMessage]] { pending }

    /// Park a message. Returns false when there was nothing to park — an empty
    /// composer must not leave a blank chip the user then has to delete.
    @discardableResult
    mutating func enqueue(_ message: QueuedMessage, for session: UUID) -> Bool {
        var trimmed = message
        trimmed.text = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasContent else { return false }
        pending[session, default: []].append(trimmed)
        return true
    }

    mutating func remove(_ id: UUID, from session: UUID) {
        guard var list = pending[session] else { return }
        list.removeAll { $0.id == id }
        pending[session] = list.isEmpty ? nil : list
    }

    mutating func clear(_ session: UUID) { pending[session] = nil }

    /// Take everything pending for `session` as ONE message, emptying the queue.
    ///
    /// Combined rather than delivered one per boundary: a queue is a steer, and
    /// a steer split across N rounds lets the model act on half of it before it
    /// has read the rest.
    mutating func drain(_ session: UUID) -> QueuedMessage? {
        let items = messages(for: session)
        pending[session] = nil
        return Self.combined(items)
    }

    /// Join queued messages into the single message the conversation receives.
    /// Blank-line separated so each keystroke-session still reads as its own
    /// paragraph; blank texts are skipped (a captionless image between two typed
    /// messages must not open a hole of empty lines) while its attachments ride
    /// along. Absent attachments stay nil, never `[]` — the turn sites downstream
    /// branch on nil.
    static func combined(_ items: [QueuedMessage]) -> QueuedMessage? {
        guard !items.isEmpty else { return nil }
        let text = items.map(\.text)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: "\n\n")
        let images = items.flatMap { $0.images ?? [] }
        let audio = items.flatMap { $0.audio ?? [] }
        return QueuedMessage(text: text,
                             images: images.isEmpty ? nil : images,
                             audio: audio.isEmpty ? nil : audio)
    }
}

/// What a composer submission MEANS. One answer for the Return key and the Send
/// button: while this chat is generating, a submission parks the message instead
/// of being swallowed (the old behaviour) or superseding the running turn.
enum ComposerSubmitAction: Equatable {
    case send
    case queue
    case ignore

    static func resolve(generating: Bool, serverRunning: Bool, hasContent: Bool) -> ComposerSubmitAction {
        // A queued message is a promise to deliver it at the end of a turn.
        // With the server down there is no turn to end, so parking one would be
        // a promise nothing can keep.
        guard serverRunning, hasContent else { return .ignore }
        return generating ? .queue : .send
    }
}
