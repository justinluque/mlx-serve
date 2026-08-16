import XCTest
@testable import MLXCore

/// `PermissionMode` is the interactive chat's half of the security surface that
/// `ApprovalPolicyTests` pins for unattended runs, so it gets the same treatment.
///
/// Two properties matter more than any individual cell:
///
/// 1. **`.ask` is byte-identical to what the chat did before modes existed** —
///    a sheet for every tool call except the `searchDocuments` carve-out. It is
///    the default, so an existing user who never opens the picker sees no
///    change. That is the upgrade guarantee and it has its own test.
/// 2. **The three permissive modes DELEGATE** rather than re-implement. The
///    delegation is pinned against `ApprovalPolicy.decide` across a matrix, so
///    the chat and a task run can never drift into two different answers for
///    the same question — which is the entire reason this enum routes through
///    the existing policy instead of adding a second one.
final class PermissionModeTests: XCTestCase {

    private let wd = "/tmp/chat-folder"

    private func decide(_ tool: String, _ mode: PermissionMode,
                        args: [String: String] = [:],
                        wd: String? = "/tmp/chat-folder") -> ApprovalDecision {
        mode.decide(tool: tool, arguments: args, rawArguments: "", workingDirectory: wd)
    }

    /// Every tool name that appears in the app's own definitions, plus a couple
    /// of shapes the model invents, so a new mode can't quietly leave a hole.
    private let everyTool = [
        "readFile", "writeFile", "editFile", "searchFiles", "listFiles", "shell",
        "browse", "webSearch", "saveMemory", "cwd", "createTask",
        "generate_image", "server__someTool", "frobnicate",
    ]

    // MARK: - The upgrade guarantee

    func testDefaultIsAsk() {
        XCTAssertEqual(PermissionMode.default, .ask,
                       "Changing the default silently loosens every existing install.")
    }

    func testAskPromptsForEveryToolIncludingReadOnlyOnes() {
        // The chat's old `requestToolApproval` asked for EVERYTHING except
        // searchDocuments — including readFile. `.ask` must not quietly adopt
        // ApprovalPolicy's readOnly allowances, which are a LOOSER rule.
        for tool in everyTool {
            guard case .ask = decide(tool, .ask, args: ["path": "notes.txt"]) else {
                return XCTFail("`.ask` must prompt for \(tool)")
            }
        }
    }

    func testSearchDocumentsNeverInterruptsInAnyMode() {
        // Its gate is the attached folder, which is stronger than a toggle.
        for mode in PermissionMode.allCases {
            XCTAssertEqual(decide("searchDocuments", mode), .allow,
                           "\(mode) should never interrupt for searchDocuments")
        }
    }

    func testTheCarveOutIsNotAddedToApprovalPolicysReadOnlySet() {
        // Adding it there would loosen unattended task runs, which never asked
        // for the carve-out. It belongs to this layer only.
        XCTAssertFalse(ApprovalPolicy.readOnlyTools.contains("searchDocuments"))
    }

    // MARK: - Delegation (the anti-drift property)

    func testAutonomyIsNonNilForExactlyTheDelegatingModes() {
        XCTAssertNil(PermissionMode.plan.autonomy)
        XCTAssertNil(PermissionMode.ask.autonomy)
        XCTAssertEqual(PermissionMode.acceptEdits.autonomy, .workspace)
        XCTAssertEqual(PermissionMode.auto.autonomy, .fullAuto)
        XCTAssertEqual(PermissionMode.bypass.autonomy, .yolo)
    }

    func testDelegatingModesMatchApprovalPolicyExactly() {
        let paths = ["notes.txt", "/etc/passwd", "sub/dir/file.swift", ""]
        for mode in PermissionMode.allCases {
            guard let autonomy = mode.autonomy else { continue }
            for tool in everyTool {
                for path in paths {
                    for workdir in [wd, nil] {
                        let mine = mode.decide(tool: tool, arguments: ["path": path],
                                               rawArguments: "", workingDirectory: workdir)
                        let theirs = ApprovalPolicy.decide(
                            tool: tool, autonomy: autonomy, arguments: ["path": path],
                            rawArguments: "", workingDirectory: workdir)
                        XCTAssertEqual(mine, theirs,
                                       "\(mode)/\(tool)/path=\(path)/wd=\(workdir ?? "nil") drifted from ApprovalPolicy")
                    }
                }
            }
        }
    }

    // MARK: - Plan mode REFUSES, it does not ask

    func testPlanAllowsReadOnlyTools() {
        for tool in ApprovalPolicy.readOnlyTools {
            XCTAssertEqual(decide(tool, .plan), .allow)
        }
    }

    func testPlanDeniesMutationsRatherThanAskingAboutThem() {
        // This is what separates Plan from `.readOnly` autonomy, which ASKS.
        // A prompt would defeat the point: plan mode exists so a run can be
        // explored without a single interruption.
        for tool in ["writeFile", "editFile", "shell", "saveMemory", "createTask", "frobnicate"] {
            guard case .deny = decide(tool, .plan, args: ["path": "notes.txt"]) else {
                return XCTFail("plan mode should DENY \(tool), not ask")
            }
        }
    }

    func testPlanDenialTellsTheModelToStopRetryingAndWriteAPlan() {
        // An error echo teaches the model the error — a bare "denied" is what
        // makes a model retry the same call until the loop budget runs out.
        guard case .deny(let reason) = decide("shell", .plan) else {
            return XCTFail("expected a denial")
        }
        XCTAssertTrue(reason.contains("shell"), "the refusal must name the tool")
        XCTAssertTrue(reason.lowercased().contains("plan"),
                      "the refusal must name the mode so the user can act on it")
        XCTAssertTrue(reason.lowercased().contains("not retry")
                        || reason.lowercased().contains("don't retry"),
                      "the refusal must stop the retry loop")
    }

    func testBypassAllowsEverythingEverywhere() {
        for tool in everyTool {
            XCTAssertEqual(decide(tool, .bypass, args: ["path": "/etc/passwd"], wd: nil), .allow)
        }
    }

    // MARK: - Escapes still stop the two middle modes

    func testAcceptEditsAndAutoStillAskWhenAWriteLeavesTheWorkspace() {
        for mode in [PermissionMode.acceptEdits, .auto] {
            for tool in ["writeFile", "editFile"] {
                guard case .ask = decide(tool, mode, args: ["path": "/etc/passwd"]) else {
                    return XCTFail("\(mode) must ask before \(tool) escapes the workspace")
                }
                XCTAssertEqual(decide(tool, mode, args: ["path": "inside.txt"]), .allow)
            }
        }
    }

    func testAcceptEditsStillAsksForShellWhileAutoDoesNot() {
        // The one behavioural difference between the two middle modes — if this
        // collapses, one of them is a dead menu entry.
        guard case .ask = decide("shell", .acceptEdits) else {
            return XCTFail("acceptEdits should ask before running a command")
        }
        XCTAssertEqual(decide("shell", .auto), .allow)
    }

    // MARK: - Persistence + UI contract

    func testRawValuesAreStableAcrossReleases() {
        // Stored on the session and in ServerOptions; renaming one silently
        // resets every user who had it selected.
        XCTAssertEqual(PermissionMode.plan.rawValue, "plan")
        XCTAssertEqual(PermissionMode.ask.rawValue, "ask")
        XCTAssertEqual(PermissionMode.acceptEdits.rawValue, "acceptEdits")
        XCTAssertEqual(PermissionMode.auto.rawValue, "auto")
        XCTAssertEqual(PermissionMode.bypass.rawValue, "bypass")
    }

    func testAnUnknownStoredModeFallsBackToTheDefaultInsteadOfFailing() {
        // Same rule as the session's disabledTools raw strings: a retired mode
        // must leave an unknown name, never break the whole decode.
        XCTAssertEqual(PermissionMode(stored: "yolo-supreme"), .ask)
        XCTAssertEqual(PermissionMode(stored: nil), .ask)
        XCTAssertEqual(PermissionMode(stored: "auto"), .auto)
    }

    func testCaseOrderRunsLeastToMostPermissive() {
        // Case order IS menu order (same convention as SettingsCategory), and a
        // permission menu that isn't ordered by power is a misclick generator.
        XCTAssertEqual(PermissionMode.allCases, [.plan, .ask, .acceptEdits, .auto, .bypass])
    }

    func testEveryModeHasDistinctUserFacingText() {
        // A mode nobody can tell apart in the menu is a mode nobody picks
        // correctly — and `summary` is what the pill's hover card shows.
        let titles = PermissionMode.allCases.map(\.title)
        let summaries = PermissionMode.allCases.map(\.summary)
        XCTAssertEqual(Set(titles).count, PermissionMode.allCases.count, "duplicate titles")
        XCTAssertEqual(Set(summaries).count, PermissionMode.allCases.count, "duplicate summaries")
        for mode in PermissionMode.allCases {
            XCTAssertFalse(mode.title.isEmpty)
            XCTAssertFalse(mode.icon.isEmpty)
            // Capped at a glance, same bar as ComposerTip bodies.
            XCTAssertLessThanOrEqual(mode.summary.count, 120, "\(mode) summary is too long to read")
        }
    }

    func testOnlyBypassIsMarkedAsUnguarded() {
        // Drives the warning tint. If another mode ever reads as unguarded the
        // tint stops meaning anything.
        XCTAssertEqual(PermissionMode.allCases.filter(\.isUnguarded), [.bypass])
    }

    // MARK: - Agent compatibility

    func testAgentAutoApproveMapsOntoModesWithoutChangingBehaviour() {
        // `autoApproveTools == true` meant "never ask" before modes existed, so
        // it must land on bypass — anything softer silently starts prompting
        // users whose agents were configured not to.
        XCTAssertEqual(PermissionMode.forAgentAutoApprove(true, default: .ask), .bypass)
        XCTAssertEqual(PermissionMode.forAgentAutoApprove(false, default: .auto), .ask)
        XCTAssertEqual(PermissionMode.forAgentAutoApprove(nil, default: .acceptEdits), .acceptEdits)
    }
}
