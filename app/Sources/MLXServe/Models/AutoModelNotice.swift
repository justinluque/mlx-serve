import Foundation

/// The transcript's record of the app choosing a model for you.
///
/// A cold-start send loads a multi-GB checkpoint and changes which model the
/// whole app is pointed at. Doing that silently is the class of thing you only
/// discover later, from a pill you weren't looking at — so the pick gets a row
/// that names the model and the reason, in the same place the answer will
/// appear. Data on the message, never text inside it: `content` rides back to
/// the model as history, and a banner glued into it is read as prose the
/// assistant wrote (the truncation-notice rule).
struct AutoModelNotice: Codable, Equatable {

    enum Kind: String, Codable {
        /// The checkpoint is loading; the turn runs when it's up.
        case loading
        /// Loaded, and the turn is under way.
        case loaded
        /// Nothing downloaded can do what the message needs. The row carries a
        /// Download button and the message waits for it.
        case needsDownload
        /// The load failed — the message was NOT sent.
        case failed
    }

    var kind: Kind
    var modelName: String
    var reason: AutoModelReason
    /// The repo to fetch, on `.needsDownload`.
    var repoId: String? = nil
    /// The message waiting to be sent, shown while a download runs so the text
    /// you typed is on screen rather than swallowed by the composer.
    var queuedPreview: String? = nil
    /// Whatever the server said, on `.failed`. Absent is normal — a timeout has
    /// nothing to quote.
    var failureMessage: String? = nil

    var headline: String {
        switch kind {
        case .loading:       return "Starting \(modelName)…"
        case .loaded:        return "Answering with \(modelName)"
        case .needsDownload: return "\(modelName) can handle this"
        case .failed:        return "Couldn't start \(modelName)"
        }
    }

    var detail: String {
        switch kind {
        case .loading, .loaded:
            return reason.sentence(modelName: modelName)
        case .needsDownload:
            return reason.sentence(modelName: modelName)
                + " It isn't downloaded yet — your message sends as soon as it is."
        case .failed:
            // Never a guessed diagnosis: quote the server, or say plainly that
            // there is nothing to quote. Both leave the message in your hands.
            if let failureMessage, !failureMessage.isEmpty {
                return "\(failureMessage) Your message wasn't sent — try again, or pick a model yourself."
            }
            return "Your message wasn't sent — try again, or pick a model yourself."
        }
    }
}

/// Whether the composer's Send does anything, and what.
///
/// The feature is that a message is no longer refused for want of a loaded
/// model, so this is the one place that could quietly take it back: every
/// `.disabled` below is a state where sending has nowhere to go.
enum ChatSendGate: Equatable {
    /// Send the turn the normal way — a model is already answering here.
    case send
    /// Nothing is loaded: pick a model from the input, load it, then send.
    case autoPick
    case disabled

    static func resolve(status: ServerStatus,
                        hasContent: Bool,
                        hasAutoPick: Bool,
                        isAwaitingAutoModel: Bool) -> ChatSendGate {
        guard hasContent else { return .disabled }
        // A message is already queued behind a load or a download for this
        // chat; a second would race it onto a different model.
        guard !isAwaitingAutoModel else { return .disabled }
        switch status {
        case .running:
            return .send
        case .starting:
            // A load is in flight. Queueing a pick on top of it is how two
            // models end up loading at once, on one GPU.
            return .disabled
        case .stopped, .error:
            // With nothing to pick, the blocking gate sheet is already on
            // screen offering the download — a live Send with nowhere to send
            // is the dead-control class.
            return hasAutoPick ? .autoPick : .disabled
        }
    }
}
