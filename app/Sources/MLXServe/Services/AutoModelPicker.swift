import Foundation

/// Why the app chose the model it chose. The card renders `sentence(modelName:)`
/// — the copy is DATA so the wording is testable and so the transcript never
/// has to guess at a rationale it wasn't given.
///
/// Order matters: a REQUIREMENT of the message (it carries an image; it won't
/// fit) outranks a preference (tools are on), which outranks continuity (the
/// model you used last), which outranks our own guess about your Mac.
enum AutoModelReason: String, Codable, CaseIterable {
    /// The message carries an image and this model can read one.
    case vision
    /// The message is too long for the alternatives.
    case longContext
    /// Tools are on and this model is likely to call them.
    case tools
    /// It's what was selected before — the user's own choice, honoured.
    case lastUsed
    /// Nothing else decided it, so: the most capable model this Mac can run.
    case ramFit

    func sentence(modelName: String) -> String {
        switch self {
        case .vision:
            return "You attached an image, and \(modelName) can read one."
        case .longContext:
            return "This message is long, and \(modelName) has the context window for it."
        case .tools:
            return "Tools are on, and \(modelName) can call them."
        case .lastUsed:
            return "\(modelName) is the model you used last."
        case .ramFit:
            return "\(modelName) is the most capable model your Mac can run comfortably."
        }
    }
}

/// What the message being sent needs from a model, read off the composer.
///
/// Everything here is OBSERVABLE at send time — an attachment is an attachment,
/// a character count is a character count. Nothing tries to infer the topic of
/// the message: matching "sounds like code" against model NAMES is a guess that
/// only pays off for someone who happens to own a specialist, and a wrong guess
/// silently swaps the model out from under a working chat.
struct AutoModelNeed: Equatable {
    /// An image is attached; a text-only model would answer about nothing.
    var needsVision: Bool = false
    /// Rough prompt-plus-reply size, in tokens. Rough on purpose: it decides
    /// between context windows that are orders of magnitude apart.
    var estimatedTokens: Int = 0
    /// The Tools disc is on for this turn.
    var wantsTools: Bool = false

    /// Bytes-per-token is ~4 for English across every tokenizer here; images
    /// land somewhere near a few hundred tokens depending on the encoder's
    /// merge size. Both are approximations, and the reply allowance below is
    /// what keeps them from mattering: a window that exactly fits the prompt
    /// has nowhere to put the answer.
    private static let charsPerToken = 4
    private static let tokensPerImage = 400
    private static let replyAllowance = 1024

    static func from(text: String, imageCount: Int, pdfCharacters: Int, toolsEnabled: Bool) -> AutoModelNeed {
        let chars = text.count + pdfCharacters
        return AutoModelNeed(
            needsVision: imageCount > 0,
            estimatedTokens: chars / charsPerToken + imageCount * tokensPerImage + replyAllowance,
            wantsTools: toolsEnabled)
    }
}

/// What to do with a message sent while nothing is loaded.
enum AutoModelPick: Equatable {
    /// Load this local checkpoint, then run the turn.
    case use(path: String, name: String, reason: AutoModelReason)
    /// Nothing on disk can do what the message REQUIRES; offer this repo.
    case download(repoId: String, name: String, reason: AutoModelReason)
    /// No chat model at all. The blocking gate sheet owns this case — it is
    /// the screen with the starter download on it — so the picker declines
    /// rather than shipping a second answer to the same question.
    case noneAvailable
}

/// Chooses the model for a message sent while the server is down or no model is
/// selected, so the composer can accept input instead of refusing it.
///
/// Pure, and deliberately a rule scorer rather than a model call: asking a model
/// which model to load means loading one to decide — two loads and tens of
/// seconds on a single serialized GPU before the first token. Every signal it
/// needs (attachments, length, the Tools disc, and each checkpoint's own
/// config-derived facts) is already in hand synchronously.
///
/// Scope is COLD START only. Once a model is running nothing here fires — a
/// picker that switched models mid-conversation would change the voice
/// answering you between one message and the next.
enum AutoModelPicker {

    /// Weights need roughly their own size plus KV cache and runtime buffers,
    /// and the Mac has to keep running everything else. Same ×1.2 the download
    /// surfaces quote (`HFModel.ramEstimate`), against 80% of physical memory.
    private static let runtimeOverhead = 1.2
    private static let usableMemoryFraction = 0.8

    static func pick(need: AutoModelNeed,
                     candidates: [LocalModel],
                     physicalMemoryBytes: UInt64,
                     lastUsedPath: String) -> AutoModelPick {
        // Only ever offer something the server can serve chat from: drafters,
        // media checkpoints and encoders are real files the browser lists, and
        // loading one as the chat model can only fail.
        let pickable = candidates.filter(\.isChatPickable)

        // Vision is the one HARD requirement a download can fix. Answering an
        // attached image on a text-only model produces confident fiction, so
        // offer a model that can see instead.
        var pool = pickable
        var reason: AutoModelReason = .ramFit
        if need.needsVision {
            let seers = pool.filter(\.hasVision)
            guard !seers.isEmpty else {
                let offer = RecommendedModelPick.visionPick(physicalMemoryBytes: physicalMemoryBytes)
                return .download(repoId: offer.repoId, name: offer.name, reason: .vision)
            }
            pool = seers
            reason = .vision
        }

        guard !pool.isEmpty else { return .noneAvailable }

        // Context. A model whose window can't hold the message answers a
        // truncated version of it, so prefer the ones that fit — and when none
        // do, the largest window is the least-bad answer rather than a refusal.
        let holds = pool.filter { ($0.contextLength ?? Int.max) >= need.estimatedTokens }
        if holds.isEmpty {
            let widest = pool.max { ($0.contextLength ?? 0) < ($1.contextLength ?? 0) }!
            return .use(path: widest.path, name: widest.name, reason: .longContext)
        }
        if holds.count < pool.count { reason = reason == .vision ? .vision : .longContext }
        pool = holds

        // Continuity beats our ranking: a model the user selected themselves is
        // what they expect to answer, including one bigger than we'd have
        // chosen for this Mac. It only applies once the message's hard
        // requirements are satisfied — that's what surviving the filters means.
        if !lastUsedPath.isEmpty, let previous = pool.first(where: { $0.path == lastUsedPath }) {
            return .use(path: previous.path, name: previous.name,
                        reason: reason == .ramFit ? .lastUsed : reason)
        }

        let fits = pool.filter { fitsMemory($0, physicalMemoryBytes: physicalMemoryBytes) }
        guard !fits.isEmpty else {
            // Nothing fits: the smallest is the one with a chance of loading.
            // Refusing here would put the user back where this feature started.
            let smallest = pool.min { $0.sizeBytes < $1.sizeBytes }!
            return .use(path: smallest.path, name: smallest.name, reason: reason)
        }

        // Tool calling is a PREFERENCE, so it only decides a choice nothing
        // else has decided, and only when it actually narrows the field.
        if need.wantsTools, reason == .ramFit {
            let callers = fits.filter(\.hasToolCalling)
            if !callers.isEmpty, callers.count < fits.count {
                let winner = best(of: callers)
                return .use(path: winner.path, name: winner.name, reason: .tools)
            }
        }
        let winner = best(of: fits)
        return .use(path: winner.path, name: winner.name, reason: reason)
    }

    /// Bigger is better among models that fit — weight bytes are the closest
    /// thing to a capability ordering we can read off disk without a benchmark,
    /// and every alternative (a parameter count parsed out of a name, quant
    /// bits) is a subset of what the byte count already reflects. Name is the
    /// tie-break so the choice is stable across scans rather than at the mercy
    /// of directory order.
    private static func best(of models: [LocalModel]) -> LocalModel {
        models.max { a, b in
            if a.sizeBytes != b.sizeBytes { return a.sizeBytes < b.sizeBytes }
            return a.name > b.name
        }!
    }

    private static func fitsMemory(_ model: LocalModel, physicalMemoryBytes: UInt64) -> Bool {
        // An unknown size (0) is not a reason to exclude a model — discovery
        // reports 0 for layouts it couldn't sum, and hiding those would make
        // the picker's library smaller than the picker's own list.
        guard model.sizeBytes > 0 else { return true }
        let budget = Double(physicalMemoryBytes) * usableMemoryFraction
        return Double(model.sizeBytes) * runtimeOverhead <= budget
    }
}
