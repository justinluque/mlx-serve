import SwiftUI

/// The composer's permission picker: how far this conversation's tools may go
/// before the agent stops and asks.
///
/// It lives in the composer row rather than the toolbar for the same reason
/// Think / Tools / MCP do — it configures the MESSAGE being sent, not the
/// window. Unlike those three it is **labeled**: five states cannot read from
/// colour alone, which is the whole reason the icon-only discs are booleans.
///
/// It renders only while the tool loop is actually running. A permission mode
/// on a plain-chat turn governs nothing, and the composer row is a width budget
/// — a control that cannot affect the next message doesn't get a slot in it.
struct PermissionModePill: View {
    let mode: PermissionMode
    /// Set when the tab's agent decided approval for itself
    /// (`Agent.autoApproveTools`), in which case the chat's pick is ignored by
    /// `AgentResolution` and offering a live control here would be a dead one.
    var lockedBy: String?
    let onSelect: (PermissionMode) -> Void

    private var isLocked: Bool { lockedBy != nil }

    var body: some View {
        Menu {
            if let lockedBy {
                // A locked control that silently does nothing on click is the
                // dead-control class — say who decided, exactly as the
                // Think/Tools/MCP discs do.
                Text("Set by \(lockedBy)")
            } else {
                ForEach(PermissionMode.allCases) { option in
                    Button { onSelect(option) } label: {
                        // Title, its one-line summary, and a check on the
                        // current pick. The summary is what stops someone
                        // choosing Bypass because it sounded like "Auto".
                        Label {
                            Text(option.title)
                            Text(option.summary)
                        } icon: {
                            Image(systemName: option == mode ? "checkmark" : option.icon)
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: mode.icon)
                    .font(.system(size: 11, weight: .medium))
                Text(mode.title)
                    .font(.system(size: 11, weight: .medium))
                    .lineLimit(1)
                    .fixedSize()
            }
            .foregroundStyle(tint)
            .padding(.horizontal, 8)
            .frame(height: ChatMetrics.composerControlSize)
            .background(
                Capsule().fill(tint.opacity(isLocked ? 0.06 : 0.12))
            )
            .overlay(
                // The agent-locked treatment is a dashed inset ring, never a
                // dimming: colour here already means "how permissive", and
                // dimming it would read as a sixth, weaker mode.
                Capsule().strokeBorder(tint.opacity(0.5),
                                       style: StrokeStyle(lineWidth: 1, dash: isLocked ? [3, 2] : []))
                    .opacity(isLocked || mode.isUnguarded ? 1 : 0)
            )
            .contentShape(Capsule())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(helpText)
    }

    /// Colour carries permissiveness: the one unguarded mode is the only red
    /// thing in the row, and Plan reads as inert rather than active.
    private var tint: Color {
        switch mode {
        case .plan:        return .secondary
        case .ask:         return .accentColor
        case .acceptEdits: return .teal
        case .auto:        return .orange
        case .bypass:      return .red
        }
    }

    private var helpText: String {
        if let lockedBy {
            return "\(mode.title) — \(mode.summary)\nSet by \(lockedBy); this chat's own setting doesn't apply."
        }
        return "\(mode.title) — \(mode.summary)"
    }
}
