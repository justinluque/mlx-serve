import XCTest
@testable import MLXCore

/// `read_image` reads a file off disk and hands the model-facing loop a
/// `data:image/jpeg;base64,…` marker it never puts in the tool-result TEXT
/// (see `AgentMediaInline`) — `ChatTurnEngine` is what turns the marker into
/// real vision input for the model. These tests cover the handler's own
/// contract: which files it accepts, and what it says when it can't.
final class ReadImageHandlerTests: XCTestCase {

    private func makeTempDir() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("rih-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    private func writePNG(named name: String, in dir: String) throws -> String {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 4, pixelsHigh: 4,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        let png = rep.representation(using: .png, properties: [:])!
        let path = (dir as NSString).appendingPathComponent(name)
        try png.write(to: URL(fileURLWithPath: path))
        return path
    }

    func testReadsAnImageAndReturnsTheJpegMarker() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        _ = try writePNG(named: "shot.png", in: dir)

        let output = try await ReadImageHandler().execute(
            parameters: ["path": "shot.png"], workingDirectory: dir)

        XCTAssertTrue(output.contains(AgentMediaInline.jpegDataURIMarker))
        XCTAssertTrue(output.hasPrefix("Read image: shot.png"))
        let (_, jpeg) = AgentMediaInline.splitInlineImage(output)
        XCTAssertNotNil(jpeg)
        XCTAssertNotNil(NSBitmapImageRep(data: jpeg!), "the marker payload must decode as an image")
    }

    func testMissingFileNamesThePath() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }

        do {
            _ = try await ReadImageHandler().execute(
                parameters: ["path": "nope.png"], workingDirectory: dir)
            XCTFail("must throw for a missing file")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("nope.png"))
        }
    }

    func testNonImageFileNamesTheSupportedFormats() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let path = (dir as NSString).appendingPathComponent("notes.txt")
        try "just text".write(toFile: path, atomically: true, encoding: .utf8)

        do {
            _ = try await ReadImageHandler().execute(
                parameters: ["path": "notes.txt"], workingDirectory: dir)
            XCTFail("must throw for a non-image file")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("PNG"))
        }
    }

    func testMissingPathParameterThrows() async throws {
        do {
            _ = try await ReadImageHandler().execute(parameters: [:], workingDirectory: "/tmp")
            XCTFail("must throw for a missing path parameter")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("path"))
        }
    }
}
