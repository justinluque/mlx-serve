import Foundation

/// How much the agent loop may do on its own before it stops and asks.
///
/// Before this existed the interactive chat had exactly two states: a modal
/// sheet for **every** tool call, or a per-session "Always Allow" that lived in
/// memory only (`SessionToolAllowList`) and was wiped on relaunch. That is one
/// interruption per shell command for anyone doing real work, and the only
/// escape was the least reversible one.
///
/// Five modes, ordered least to most permissive (case order IS menu order).
/// Two are decided here; the other three **delegate verbatim** to
/// `ApprovalPolicy.decide` at the matching `TaskAutonomy`. The delegation is
/// the point: the interactive chat and an unattended task run now answer the
/// same question with the same audited code, so they cannot drift into two
/// different notions of what "safe" means. Nothing new is invented for those
/// three, and `PermissionModeTests` pins them against the policy across a
/// matrix rather than restating its cells.
///
/// This is the **approval** layer only. Hard filesystem confinement is
/// independently enforced by `ToolExecutor.resolveAndConfine` — a bug here
/// still cannot write outside the workspace. Same split `ApprovalPolicy`
/// documents, and the reason `.bypass` is survivable at all.
enum PermissionMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Read and research only. Mutations are **refused**, not queued for
    /// approval — the point is a run with no interruptions at all, ending in a
    /// plan the user can act on.
    case plan
    /// Approve every call. The behaviour the chat has always had, and the
    /// default, so an install that never opens the picker is unchanged.
    case ask
    /// File edits inside the working directory run on their own; commands still
    /// ask, and a write that escapes the folder still asks.
    case acceptEdits
    /// Edits and commands both run; only a write leaving the working directory
    /// asks. The closest analogue to Claude Code's "Auto", and the mode most
    /// people who don't want to approve each call actually want.
    case auto
    /// Never asks. The escape hatch, deliberately named for what it does.
    case bypass

    var id: String { rawValue }

    /// The shipped default. `.ask` reproduces the pre-modes chat exactly — see
    /// `testDefaultIsAsk`, which exists so loosening this is a deliberate act
    /// with a failing test attached rather than a one-character edit.
    static let `default`: PermissionMode = .ask

    /// Tolerant decode for a stored value. An unrecognised name resolves to the
    /// default instead of failing, so a retired mode leaves a stale string in
    /// the session rather than breaking the whole session's decode — the same
    /// rule the session's `disabledTools` raw strings follow.
    init(stored: String?) {
        self = stored.flatMap(PermissionMode.init(rawValue:)) ?? .default
    }

    /// The `TaskAutonomy` this mode delegates to, or nil when it is decided
    /// here. Non-nil for exactly `.acceptEdits` / `.auto` / `.bypass`.
    var autonomy: TaskAutonomy? {
        switch self {
        case .plan, .ask:  return nil
        case .acceptEdits: return .workspace
        case .auto:        return .fullAuto
        case .bypass:      return .yolo
        }
    }

    /// Tools whose real gate is something stronger than a toggle, so they never
    /// interrupt in any mode. `searchDocuments` only ever reads a folder the
    /// user explicitly attached — this reproduces the carve-out
    /// `ChatView.requestToolApproval` has always had.
    ///
    /// Kept HERE rather than added to `ApprovalPolicy.readOnlyTools`: that set
    /// governs unattended task runs too, which never asked for the carve-out,
    /// and loosening a headless security surface as a side effect of a chat
    /// feature is exactly the kind of blast radius this file exists to avoid.
    static let neverInterrupts: Set<String> = ["searchDocuments"]

    /// The gate. Mirrors `ApprovalPolicy.decide`'s signature so the two are
    /// substitutable at the call site.
    func decide(tool: String,
                arguments: [String: String],
                rawArguments: String,
                workingDirectory: String?) -> ApprovalDecision {
        if Self.neverInterrupts.contains(tool) { return .allow }

        switch self {
        case .plan:
            return ApprovalPolicy.readOnlyTools.contains(tool)
                ? .allow
                : .deny(reason: Self.planDenial(tool: tool))

        case .ask:
            // Deliberately NOT `ApprovalPolicy.readOnlyTools`-aware: the old
            // chat asked before reading a file too, and silently widening that
            // under an unchanged default would be a behaviour change nobody
            // opted into. Someone who wants reads to pass freely picks a mode.
            return .ask(reason: "“\(tool)” can act on your Mac, so it needs your OK.")

        case .acceptEdits, .auto, .bypass:
            guard let autonomy else {
                return .ask(reason: "“\(tool)” needs your OK.")
            }
            return ApprovalPolicy.decide(tool: tool, autonomy: autonomy,
                                         arguments: arguments, rawArguments: rawArguments,
                                         workingDirectory: workingDirectory)
        }
    }

    /// The refusal a plan-mode mutation gets. It goes back to the model as tool
    /// output, so it is written to be *read* by the model: name the tool, name
    /// the mode (so the user reading the transcript knows which switch to
    /// flip), and kill the retry explicitly. A bare "denied" is how a model
    /// burns its whole iteration budget re-issuing the same call.
    static func planDenial(tool: String) -> String {
        "Error: Plan mode is on, so “\(tool)” did not run and nothing has been changed. "
            + "Do not retry this or any other tool that makes changes — reading, searching and browsing still work. "
            + "Investigate with those, then reply with the plan you would carry out so the user can approve it."
    }

    // MARK: - Agent compatibility

    /// Fold an agent's legacy tri-state `autoApproveTools` onto a mode.
    ///
    /// `true` maps to `.bypass`, not to `.auto`: before modes existed it meant
    /// "never ask", and mapping it anywhere softer would start prompting users
    /// whose agents were explicitly configured not to be prompted.
    static func forAgentAutoApprove(_ flag: Bool?, default fallback: PermissionMode) -> PermissionMode {
        switch flag {
        case .some(true):  return .bypass
        case .some(false): return .ask
        case .none:        return fallback
        }
    }

    // MARK: - User-facing text

    /// Menu row + pill label.
    var title: String {
        switch self {
        case .plan:        return "Plan"
        case .ask:         return "Ask"
        case .acceptEdits: return "Accept Edits"
        case .auto:        return "Auto"
        case .bypass:      return "Bypass"
        }
    }

    /// The menu row's second line and the pill's hover card. Capped at a glance
    /// (120 chars, the same bar `ComposerTip` bodies are held to) — a paragraph
    /// here is a paragraph nobody reads before picking.
    var summary: String {
        switch self {
        case .plan:        return "Research only. Edits and commands are refused, so you get a plan instead of changes."
        case .ask:         return "Approve each tool call before it runs."
        case .acceptEdits: return "File edits in the working folder run on their own. Commands still ask."
        case .auto:        return "Edits and commands run on their own. Only work outside the folder asks."
        case .bypass:      return "Nothing ever asks. Every tool call runs immediately."
        }
    }

    /// SF Symbol — free with the OS, no assets.
    var icon: String {
        switch self {
        case .plan:        return "list.bullet.rectangle"
        case .ask:         return "hand.raised"
        case .acceptEdits: return "square.and.pencil"
        case .auto:        return "bolt"
        case .bypass:      return "exclamationmark.triangle"
        }
    }

    /// True for the one mode with no guard rail left, so the pill and the menu
    /// row can carry a warning tint. Exactly one mode may be unguarded or the
    /// tint stops carrying information (`testOnlyBypassIsMarkedAsUnguarded`).
    var isUnguarded: Bool { self == .bypass }
}
