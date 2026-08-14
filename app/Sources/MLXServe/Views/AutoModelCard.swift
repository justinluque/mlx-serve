import SwiftUI

/// The transcript row for a model the app chose itself.
///
/// Deliberately quiet — this is a fact about the machinery, not an answer, so
/// it reads like the tool-call cards rather than like a message. It is also the
/// only place the download offer can appear, which is why the button lives here
/// and not in a sheet: a sheet would take over the window for something the
/// user can perfectly well leave running while they keep typing.
///
/// Actions arrive as closures rather than by reading the environment, because
/// this view renders inside `MessageBubble`, which the task-run viewer also
/// uses — a view that reads `@EnvironmentObject` from a host that doesn't
/// inject it traps at first render (the sheet-environment class in app/CLAUDE.md).
struct AutoModelCard: View {
    let notice: AutoModelNotice
    /// Live transfer for `notice.repoId`, when one is running. nil covers both
    /// "not started" and "finished".
    var downloadProgress: Double? = nil
    /// nil on read-only surfaces, which have no server to start — the row then
    /// renders as a record of what happened, with no dead button.
    var onDownload: (() -> Void)? = nil
    var onCancel: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 13))
                .foregroundStyle(tint)
                .frame(width: 16)

            VStack(alignment: .leading, spacing: 4) {
                Text(notice.headline)
                    .font(.caption.weight(.semibold))
                Text(notice.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                // The message you typed, while it waits. The composer has
                // already been cleared, so without this the text is nowhere.
                if let queued = notice.queuedPreview, !queued.isEmpty {
                    Text(queued)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.5)))
                }

                action
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(tint.opacity(0.08)))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private var action: some View {
        if notice.kind == .needsDownload {
            if let downloadProgress {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: downloadProgress)
                        .frame(maxWidth: 260)
                    if let onCancel {
                        Button("Cancel", action: onCancel)
                            .buttonStyle(.plain)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.top, 2)
            } else if let onDownload {
                Button("Download", action: onDownload)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .padding(.top, 2)
            }
        }
    }

    private var icon: String {
        switch notice.kind {
        case .loading:       return "arrow.triangle.2.circlepath"
        case .loaded:        return "sparkles"
        case .needsDownload: return "arrow.down.circle"
        case .failed:        return "exclamationmark.triangle"
        }
    }

    /// Colour is the whole status signal on a row this small. Red is reserved
    /// for the one state that needs acting on, exactly as the composer's Start
    /// control reserves it (`ChatServerStartControl`).
    private var tint: Color {
        notice.kind == .failed ? .red : .accentColor
    }
}
