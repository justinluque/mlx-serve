//! Ideogram 4's structured-caption layer: the magic-prompt rewriter and the
//! caption verifier, ported from `ideogram-oss/ideogram4` (Apache-2.0).
//!
//! Ideogram 4 was trained EXCLUSIVELY on structured JSON captions — bounding
//! boxes, colour palettes, per-element descriptions. A bare sentence is not a
//! weaker prompt for it, it is an out-of-distribution one, which is why the
//! reference pipeline ships a rewriter rather than treating it as an optional
//! nicety. The reference sends the rewrite to OpenRouter or to Ideogram's own
//! API; mlx-serve sends nothing anywhere, so it runs the same system prompt
//! through a chat model this server already has loaded.
//!
//! Everything here is PURE: section parsing, message building, "is this
//! already a caption", fence stripping, and verification. `server.zig` owns
//! the one impure part — actually running the chat model.

const std = @import("std");

/// The reference's `magic_prompt_system_prompts/v1.txt`, vendored verbatim.
/// Sections are `[META]`, `[SYSTEM]`, `[USER]`.
pub const system_prompt_v1 = @embedFile("data/ideogram4_magic_prompt_v1.txt");

// ── Section parsing ───────────────────────────────────────────────────────

pub const Sections = struct {
    meta: []const u8 = "",
    system: []const u8 = "",
    user: []const u8 = "",
};

/// Split a system-prompt file into its `[NAME]` blocks. A marker is a line
/// that is exactly `[` + a name with no spaces + `]`; anything else is body
/// text. Bodies come back with surrounding whitespace trimmed.
///
/// The result BORROWS from `raw` — no allocation, which is what lets the
/// embedded file be parsed at comptime or per request without a copy.
pub fn parseSections(raw: []const u8) Sections {
    var out: Sections = .{};
    var current: []const u8 = "";
    var start: usize = 0;
    var it = std.mem.splitScalar(u8, raw, '\n');
    var pos: usize = 0;
    while (it.next()) |line| {
        const line_start = pos;
        pos += line.len + 1;
        const stripped = std.mem.trim(u8, line, " \t\r");
        const is_marker = stripped.len >= 3 and
            stripped[0] == '[' and stripped[stripped.len - 1] == ']' and
            std.mem.indexOfScalar(u8, stripped, ' ') == null;
        if (!is_marker) continue;
        if (current.len != 0) assign(&out, current, trimBody(raw[start..line_start]));
        current = stripped[1 .. stripped.len - 1];
        start = pos;
    }
    if (current.len != 0) assign(&out, current, trimBody(raw[start..]));
    return out;
}

fn trimBody(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

fn assign(out: *Sections, name: []const u8, body: []const u8) void {
    if (eqlIgnoreCase(name, "meta")) out.meta = body;
    if (eqlIgnoreCase(name, "system")) out.system = body;
    if (eqlIgnoreCase(name, "user")) out.user = body;
}

fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (std.ascii.toLower(x) != std.ascii.toLower(y)) return false;
    }
    return true;
}

/// Reduce `width`×`height` to a `"W:H"` string (caller frees).
pub fn aspectRatio(allocator: std.mem.Allocator, width: u32, height: u32) ![]u8 {
    const g = @max(std.math.gcd(width, height), 1);
    return std.fmt.allocPrint(allocator, "{d}:{d}", .{ width / g, height / g });
}

/// The user message: the `[USER]` template with `{{aspect_ratio}}` and
/// `{{original_prompt}}` substituted. A template with no `{{original_prompt}}`
/// placeholder (or no `[USER]` block at all) gets the prompt appended after a
/// default framing line — the reference's own fallback. Caller frees.
pub fn buildUserMessage(allocator: std.mem.Allocator, sections: Sections, prompt: []const u8, aspect: []const u8) ![]u8 {
    const template = if (sections.user.len != 0)
        sections.user
    else
        "TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height).";
    const with_aspect = try std.mem.replaceOwned(u8, allocator, template, "{{aspect_ratio}}", aspect);
    defer allocator.free(with_aspect);
    if (std.mem.indexOf(u8, with_aspect, "{{original_prompt}}") != null) {
        return std.mem.replaceOwned(u8, allocator, with_aspect, "{{original_prompt}}", prompt);
    }
    return std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ with_aspect, prompt });
}

// ── Rewriter output handling ──────────────────────────────────────────────

/// Drop a surrounding ```-fence if the model added one. Returns a slice of the
/// input — the caller keeps ownership of the buffer.
pub fn stripCodeFences(text: []const u8) []const u8 {
    const t = std.mem.trim(u8, text, " \t\r\n");
    if (!std.mem.startsWith(u8, t, "```")) return t;
    // Drop the opening fence line (```json and friends) and a closing ``` line.
    const first_nl = std.mem.indexOfScalar(u8, t, '\n') orelse return t;
    var body = t[first_nl + 1 ..];
    const trimmed = std.mem.trimEnd(u8, body, " \t\r\n");
    if (std.mem.endsWith(u8, trimmed, "```")) {
        body = trimmed[0 .. trimmed.len - 3];
    }
    return std.mem.trim(u8, body, " \t\r\n");
}

/// True when a prompt is ALREADY a structured caption and must be passed
/// through untouched.
///
/// Cheap and deliberately shallow: a leading `{`, a trailing `}`, and one of
/// the caption's own top-level keys present. A user who hand-writes a caption
/// (the documented way to drive bbox and palette control) must never have it
/// silently rewritten, and a rewriter's own output must be recognised when it
/// comes back around.
pub fn looksLikeCaption(prompt: []const u8) bool {
    const t = std.mem.trim(u8, prompt, " \t\r\n");
    if (t.len < 2 or t[0] != '{' or t[t.len - 1] != '}') return false;
    for ([_][]const u8{ "\"compositional_deconstruction\"", "\"high_level_description\"", "\"style_description\"" }) |key| {
        if (std.mem.indexOf(u8, t, key) != null) return true;
    }
    return false;
}

// ── Caption verifier ──────────────────────────────────────────────────────

pub const MAX_WARNINGS = 32;

/// Verification result. Warnings are NON-FATAL by design: the reference can
/// raise on them, but refusing to render a caption the model would happily
/// consume trades a slightly-off image for no image at all. They are logged,
/// and they are what tells the rewriter loop that its output drifted.
pub const Report = struct {
    warnings: [MAX_WARNINGS][]const u8 = undefined,
    count: usize = 0,
    /// True when the text did not parse as JSON at all.
    invalid_json: bool = false,

    pub fn items(self: *const Report) []const []const u8 {
        return self.warnings[0..self.count];
    }
    pub fn ok(self: *const Report) bool {
        return self.count == 0 and !self.invalid_json;
    }
    fn add(self: *Report, msg: []const u8) void {
        if (self.count >= MAX_WARNINGS) return;
        self.warnings[self.count] = msg;
        self.count += 1;
    }
};

const top_level_known = [_][]const u8{ "high_level_description", "style_description", "compositional_deconstruction" };
const style_known = [_][]const u8{ "aesthetics", "lighting", "photo", "art_style", "medium", "color_palette" };
const element_known = [_][]const u8{ "type", "bbox", "text", "desc", "color_palette" };
const style_order_photo = [_][]const u8{ "aesthetics", "lighting", "photo", "medium", "color_palette" };
const style_order_non_photo = [_][]const u8{ "aesthetics", "lighting", "medium", "art_style", "color_palette" };
const cd_order = [_][]const u8{ "background", "elements" };
const element_order_obj = [_][]const u8{ "type", "bbox", "desc", "color_palette" };
const element_order_text = [_][]const u8{ "type", "bbox", "text", "desc", "color_palette" };

const bbox_min: i64 = 0;
const bbox_max: i64 = 1000;
const style_palette_max: usize = 16;
const element_palette_max: usize = 5;

/// Verify a caption. `allocator` is used only for the JSON parse; every
/// warning is a static string, so the `Report` outlives the parse.
pub fn verify(allocator: std.mem.Allocator, raw: []const u8) Report {
    var rep: Report = .{};
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch {
        rep.invalid_json = true;
        rep.add("root: not valid JSON");
        return rep;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        rep.add("root: expected a JSON object");
        return rep;
    }
    const root = parsed.value.object;
    checkUnknown(&rep, root, &top_level_known, "root: unknown top-level key");

    if (root.get("high_level_description")) |v| {
        if (v != .string) rep.add("high_level_description: expected a string");
    }
    if (root.get("style_description")) |v| verifyStyle(&rep, v);
    if (root.get("compositional_deconstruction")) |v| {
        verifyComposition(&rep, v);
    } else {
        rep.add("root: 'compositional_deconstruction' must exist");
    }
    return rep;
}

fn verifyStyle(rep: *Report, v: std.json.Value) void {
    if (v != .object) {
        rep.add("style_description: expected an object");
        return;
    }
    const sd = v.object;
    checkUnknown(rep, sd, &style_known, "style_description: unknown key");
    const has_photo = sd.get("photo") != null;
    const has_art = sd.get("art_style") != null;
    if (has_photo and has_art) {
        rep.add("style_description: contains both 'photo' and 'art_style'; expected exactly one");
        return;
    }
    if (!has_photo and !has_art) {
        rep.add("style_description: expected one of 'photo' or 'art_style'");
        return;
    }
    checkOrder(rep, sd, if (has_photo) &style_order_photo else &style_order_non_photo, "style_description: keys out of order");
    if (sd.get("color_palette")) |p| verifyPalette(rep, p, style_palette_max, "style_description.color_palette");
}

fn verifyComposition(rep: *Report, v: std.json.Value) void {
    if (v != .object) {
        rep.add("compositional_deconstruction: expected an object");
        return;
    }
    const cd = v.object;
    const bg = cd.get("background") orelse {
        rep.add("compositional_deconstruction: 'background' must exist");
        return;
    };
    if (bg != .string) {
        rep.add("compositional_deconstruction.background: expected a string");
        return;
    }
    const els = cd.get("elements") orelse {
        rep.add("compositional_deconstruction: 'elements' must exist");
        return;
    };
    checkOrder(rep, cd, &cd_order, "compositional_deconstruction: keys out of order");
    if (els != .array) {
        rep.add("compositional_deconstruction.elements: expected a list");
        return;
    }
    for (els.array.items) |el| verifyElement(rep, el);
}

fn verifyElement(rep: *Report, v: std.json.Value) void {
    if (v != .object) {
        rep.add("elements[]: expected an object");
        return;
    }
    const el = v.object;
    checkUnknown(rep, el, &element_known, "elements[]: unknown key");
    const ty = el.get("type") orelse {
        rep.add("elements[]: 'type' must exist");
        return;
    };
    const is_text = ty == .string and std.mem.eql(u8, ty.string, "text");
    const is_obj = ty == .string and std.mem.eql(u8, ty.string, "obj");
    if (!is_text and !is_obj) {
        rep.add("elements[]: 'type' must be 'obj' or 'text'");
        return;
    }
    checkOrder(rep, el, if (is_text) &element_order_text else &element_order_obj, "elements[]: keys out of order");
    if (el.get("bbox")) |b| verifyBbox(rep, b);
    if (el.get("color_palette")) |p| verifyPalette(rep, p, element_palette_max, "elements[].color_palette");
}

/// bbox is `[ymin, xmin, ymax, xmax]` on a 0–1000 grid — NOT pixels and NOT
/// x-first. Both halves are checked: the range, and that each pair is ordered.
fn verifyBbox(rep: *Report, v: std.json.Value) void {
    if (v != .array or v.array.items.len != 4) {
        rep.add("elements[].bbox: expected [ymin, xmin, ymax, xmax]");
        return;
    }
    var vals: [4]i64 = undefined;
    for (v.array.items, 0..) |item, i| {
        if (item != .integer) {
            rep.add("elements[].bbox: all values must be integers");
            return;
        }
        vals[i] = item.integer;
    }
    for (vals) |x| {
        if (x < bbox_min or x > bbox_max) {
            rep.add("elements[].bbox: values must be in [0, 1000]");
            break;
        }
    }
    if (vals[0] > vals[2]) rep.add("elements[].bbox: ymin > ymax");
    if (vals[1] > vals[3]) rep.add("elements[].bbox: xmin > xmax");
}

fn verifyPalette(rep: *Report, v: std.json.Value, max_colors: usize, comptime path: []const u8) void {
    if (v != .array) {
        rep.add(path ++ ": expected a list");
        return;
    }
    if (v.array.items.len > max_colors) {
        rep.add(path ++ ": too many colors");
        return;
    }
    for (v.array.items) |c| {
        if (c != .string or !isHexColor(c.string)) {
            rep.add(path ++ ": entries must be '#RRGGBB' hex colors");
            return;
        }
    }
}

/// `#RRGGBB` with UPPERCASE hex digits, matching the reference's own check —
/// the model was trained on that spelling.
pub fn isHexColor(s: []const u8) bool {
    if (s.len != 7 or s[0] != '#') return false;
    for (s[1..]) |c| {
        const ok = (c >= '0' and c <= '9') or (c >= 'A' and c <= 'F');
        if (!ok) return false;
    }
    return true;
}

fn checkUnknown(rep: *Report, obj: std.json.ObjectMap, known: []const []const u8, comptime msg: []const u8) void {
    var it = obj.iterator();
    while (it.next()) |e| {
        var found = false;
        for (known) |k| {
            if (std.mem.eql(u8, e.key_ptr.*, k)) {
                found = true;
                break;
            }
        }
        if (!found) rep.add(msg);
    }
}

/// The caption's keys must appear in the canonical order. Only the keys that
/// are PRESENT are compared — an optional key being absent is fine, but two
/// present keys appearing swapped is not, because the model was trained on
/// one ordering and JSON key order is preserved in the text it sees.
fn checkOrder(rep: *Report, obj: std.json.ObjectMap, order: []const []const u8, comptime msg: []const u8) void {
    var last: usize = 0;
    var it = obj.iterator();
    while (it.next()) |e| {
        for (order, 0..) |k, i| {
            if (!std.mem.eql(u8, e.key_ptr.*, k)) continue;
            if (i < last) {
                rep.add(msg);
                return;
            }
            last = i;
            break;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

test "the vendored system prompt parses into its three sections" {
    const s = parseSections(system_prompt_v1);
    try testing.expect(s.meta.len > 0);
    try testing.expect(s.system.len > 0);
    try testing.expect(s.user.len > 0);
    // The [SYSTEM] block is the actual instruction set, and it must not have
    // swallowed the marker lines around it.
    try testing.expect(std.mem.indexOf(u8, s.system, "[USER]") == null);
    try testing.expect(std.mem.indexOf(u8, s.system, "[META]") == null);
    try testing.expect(std.mem.indexOf(u8, s.system, "compositional_deconstruction") != null);
    try testing.expect(std.mem.indexOf(u8, s.meta, "thinking_mode") != null);
    // The [USER] template is what carries the two substitutions.
    try testing.expect(std.mem.indexOf(u8, s.user, "{{aspect_ratio}}") != null);
}

test "section parsing ignores bracketed lines that are not markers" {
    const raw =
        \\[SYSTEM]
        \\body one
        \\[not a marker]
        \\still body
        \\[USER]
        \\hello {{original_prompt}}
    ;
    const s = parseSections(raw);
    try testing.expectEqualStrings("body one\n[not a marker]\nstill body", s.system);
    try testing.expectEqualStrings("hello {{original_prompt}}", s.user);
}

test "aspect ratios reduce by gcd" {
    const a = testing.allocator;
    for ([_]struct { w: u32, h: u32, want: []const u8 }{
        .{ .w = 1024, .h = 1024, .want = "1:1" },
        .{ .w = 1920, .h = 1080, .want = "16:9" },
        .{ .w = 1024, .h = 1280, .want = "4:5" },
        .{ .w = 1536, .h = 512, .want = "3:1" },
    }) |c| {
        const got = try aspectRatio(a, c.w, c.h);
        defer a.free(got);
        try testing.expectEqualStrings(c.want, got);
    }
}

test "the user message substitutes both placeholders, and falls back when it can't" {
    const a = testing.allocator;
    {
        const s = Sections{ .user = "RATIO: {{aspect_ratio}}\nIDEA: {{original_prompt}}" };
        const m = try buildUserMessage(a, s, "a red barn", "16:9");
        defer a.free(m);
        try testing.expectEqualStrings("RATIO: 16:9\nIDEA: a red barn", m);
    }
    {
        // No {{original_prompt}} → the prompt is appended, not dropped.
        const s = Sections{ .user = "RATIO: {{aspect_ratio}}" };
        const m = try buildUserMessage(a, s, "a red barn", "1:1");
        defer a.free(m);
        try testing.expectEqualStrings("RATIO: 1:1\n\na red barn", m);
    }
    {
        // No [USER] block at all → the reference's default framing line.
        const m = try buildUserMessage(a, .{}, "a red barn", "4:5");
        defer a.free(m);
        try testing.expect(std.mem.startsWith(u8, m, "TARGET IMAGE ASPECT RATIO: 4:5"));
        try testing.expect(std.mem.endsWith(u8, m, "a red barn"));
    }
}

test "the real v1 template puts the prompt and the ratio into the user message" {
    const a = testing.allocator;
    const s = parseSections(system_prompt_v1);
    const m = try buildUserMessage(a, s, "a lighthouse at dusk", "16:9");
    defer a.free(m);
    try testing.expect(std.mem.indexOf(u8, m, "a lighthouse at dusk") != null);
    try testing.expect(std.mem.indexOf(u8, m, "16:9") != null);
    try testing.expect(std.mem.indexOf(u8, m, "{{") == null);
}

test "code fences come off, fenced or not" {
    try testing.expectEqualStrings("{\"a\":1}", stripCodeFences("```json\n{\"a\":1}\n```"));
    try testing.expectEqualStrings("{\"a\":1}", stripCodeFences("```\n{\"a\":1}\n```"));
    try testing.expectEqualStrings("{\"a\":1}", stripCodeFences("  {\"a\":1}  "));
    // A fence with no closing line still yields the body.
    try testing.expectEqualStrings("{\"a\":1}", stripCodeFences("```json\n{\"a\":1}"));
}

test "a hand-written caption is passed through, a sentence is not" {
    try testing.expect(looksLikeCaption(
        \\{"high_level_description":"x","compositional_deconstruction":{"background":"y","elements":[]}}
    ));
    try testing.expect(!looksLikeCaption("a red barn at sunset"));
    // JSON, but not a caption — rewriting it is still the right call.
    try testing.expect(!looksLikeCaption("{\"foo\": 1}"));
    // Not JSON at all, even though it mentions the key.
    try testing.expect(!looksLikeCaption("make a compositional_deconstruction of a barn"));
}

test "a well-formed caption verifies clean" {
    const caption =
        \\{"high_level_description":"A red barn.",
        \\ "style_description":{"aesthetics":"warm","lighting":"golden hour","photo":"35mm","medium":"photograph","color_palette":["#C0392B","#F1C40F"]},
        \\ "compositional_deconstruction":{"background":"a wheat field","elements":[
        \\   {"type":"obj","bbox":[100,200,800,900],"desc":"a red barn","color_palette":["#C0392B"]},
        \\   {"type":"text","bbox":[10,10,90,400],"text":"HARVEST","desc":"painted sign","color_palette":["#FFFFFF"]}]}}
    ;
    var rep = verify(testing.allocator, caption);
    if (!rep.ok()) {
        for (rep.items()) |w| std.debug.print("unexpected warning: {s}\n", .{w});
    }
    try testing.expect(rep.ok());
}

test "the verifier catches every failure the reference names" {
    const a = testing.allocator;
    const cases = [_]struct { caption: []const u8, want: []const u8 }{
        // Not JSON at all.
        .{ .caption = "a red barn", .want = "not valid JSON" },
        // The one mandatory top-level key.
        .{ .caption = "{\"high_level_description\":\"x\"}", .want = "'compositional_deconstruction' must exist" },
        // Exactly one of photo/art_style.
        .{
            .caption =
            \\{"style_description":{"aesthetics":"a","lighting":"b","photo":"c","art_style":"d"},"compositional_deconstruction":{"background":"x","elements":[]}}
            ,
            .want = "both 'photo' and 'art_style'",
        },
        .{
            .caption =
            \\{"style_description":{"aesthetics":"a"},"compositional_deconstruction":{"background":"x","elements":[]}}
            ,
            .want = "expected one of 'photo' or 'art_style'",
        },
        // bbox is [ymin, xmin, ymax, xmax] on a 0–1000 grid.
        .{
            .caption =
            \\{"compositional_deconstruction":{"background":"x","elements":[{"type":"obj","bbox":[0,0,2000,10],"desc":"d"}]}}
            ,
            .want = "values must be in [0, 1000]",
        },
        .{
            .caption =
            \\{"compositional_deconstruction":{"background":"x","elements":[{"type":"obj","bbox":[900,0,100,10],"desc":"d"}]}}
            ,
            .want = "ymin > ymax",
        },
        .{
            .caption =
            \\{"compositional_deconstruction":{"background":"x","elements":[{"type":"obj","bbox":[0,900,10,100],"desc":"d"}]}}
            ,
            .want = "xmin > xmax",
        },
        .{
            .caption =
            \\{"compositional_deconstruction":{"background":"x","elements":[{"type":"obj","bbox":[0,0,10],"desc":"d"}]}}
            ,
            .want = "expected [ymin, xmin, ymax, xmax]",
        },
        // Element type vocabulary.
        .{
            .caption =
            \\{"compositional_deconstruction":{"background":"x","elements":[{"type":"shape","desc":"d"}]}}
            ,
            .want = "'type' must be 'obj' or 'text'",
        },
        // Palette shape and size.
        .{
            .caption =
            \\{"compositional_deconstruction":{"background":"x","elements":[{"type":"obj","desc":"d","color_palette":["red"]}]}}
            ,
            .want = "hex colors",
        },
        .{
            .caption =
            \\{"compositional_deconstruction":{"background":"x","elements":[{"type":"obj","desc":"d","color_palette":["#000000","#000000","#000000","#000000","#000000","#000000"]}]}}
            ,
            .want = "too many colors",
        },
        // Unknown keys, at both levels.
        .{
            .caption =
            \\{"vibe":"x","compositional_deconstruction":{"background":"x","elements":[]}}
            ,
            .want = "unknown top-level key",
        },
        // Key ORDER — the model saw one ordering in training.
        .{
            .caption =
            \\{"compositional_deconstruction":{"background":"x","elements":[{"bbox":[0,0,10,10],"type":"obj","desc":"d"}]}}
            ,
            .want = "keys out of order",
        },
    };
    for (cases, 0..) |c, i| {
        var rep = verify(a, c.caption);
        var found = false;
        for (rep.items()) |w| {
            if (std.mem.indexOf(u8, w, c.want) != null) found = true;
        }
        if (!found) {
            std.debug.print("case {d}: no warning containing '{s}'; got {d}:\n", .{ i, c.want, rep.count });
            for (rep.items()) |w| std.debug.print("  - {s}\n", .{w});
        }
        try testing.expect(found);
    }
}

test "hex colors are uppercase #RRGGBB, matching what the model was trained on" {
    try testing.expect(isHexColor("#C0392B"));
    try testing.expect(isHexColor("#000000"));
    try testing.expect(!isHexColor("#c0392b")); // lowercase
    try testing.expect(!isHexColor("#FFF")); // short form
    try testing.expect(!isHexColor("C0392B")); // no hash
    try testing.expect(!isHexColor("#GGGGGG"));
}
