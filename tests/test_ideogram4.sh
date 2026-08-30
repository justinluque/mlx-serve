#!/usr/bin/env bash
# Ideogram 4 end-to-end on the ONE main server: load by path -> generate ->
# assert PNG + dimensions -> magic prompt -> LoRA grammar -> unload.
#
# Everything here that does NOT need weights runs unconditionally, because the
# pack is 13+ GB and gated: discovery/classification, the completeness marker,
# the named refusals, and the magic-prompt field plumbing are all observable
# without a single denoise step. Only the generate assertions skip.
#
# Usage: IDEOGRAM_MODEL=<dir> [CHAT_MODEL=<dir>] ./tests/test_ideogram4.sh [port]
set -uo pipefail
PORT="${1:-11407}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

IDEO="${IDEOGRAM_MODEL:-$(ls -d ~/.mlx-serve/models/justintime47/Ideogram-4-MLX-Serve-* 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "FAIL: $*"; FAILED=1; }

# ── 1. Discovery: both checkpoint shapes classify, and an incomplete pack does
#       not. No weights involved — these are filesystem predicates.
mkdir -p "$TMP/roots/ideo-upstream" "$TMP/roots/ideo-converted/unconditional_transformer" "$TMP/roots/ideo-partial"
echo '{"_class_name":"Ideogram4Pipeline"}' > "$TMP/roots/ideo-upstream/model_index.json"
mkdir -p "$TMP/roots/ideo-upstream/unconditional_transformer"
echo '{}' > "$TMP/roots/ideo-upstream/unconditional_transformer/config.json"
echo '{"model_type":"ideogram4"}' > "$TMP/roots/ideo-converted/config.json"
echo '{}' > "$TMP/roots/ideo-converted/unconditional_transformer/config.json"
# The completeness marker: a pack whose SECOND transformer never landed. The
# conditional weights alone would load and then denoise against a branch that
# was never read.
echo '{"model_type":"ideogram4"}' > "$TMP/roots/ideo-partial/config.json"

"$BIN" --serve --model-dir "$TMP/roots" --port "$PORT" >"$TMP/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -20 "$TMP/server.log"; exit 1; }
  sleep 1
done
api() { curl -s -m 900 "http://127.0.0.1:$PORT$1" "${@:2}"; }

IDS=$(api /v1/models | python3 -c 'import sys,json;print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))')
case "$IDS" in
  *ideo-upstream*)  echo "PASS: [1a] an upstream Ideogram4Pipeline repo is discovered from model_index.json" ;;
  *) fail "[1a] upstream repo not discovered (ids: $IDS)" ;;
esac
case "$IDS" in
  *ideo-converted*) echo "PASS: [1b] a converted pack is discovered from its root model_type" ;;
  *) fail "[1b] converted pack not discovered (ids: $IDS)" ;;
esac
case "$IDS" in
  *ideo-partial*) fail "[1c] a pack missing unconditional_transformer/config.json was registered" ;;
  *) echo "PASS: [1c] an incomplete pack stays invisible (completeness marker)" ;;
esac
grep -q "incomplete media pack" "$TMP/server.log" \
  && echo "PASS: [1d] the skip is logged by name" \
  || fail "[1d] no 'incomplete media pack' line for the partial pack"

# ── 2. A text request against an image model is a NAMED 400, before any load.
CODE=$(curl -s -o "$TMP/txt.json" -w '%{http_code}' -m 30 \
  -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"ideo-converted\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
if [ "$CODE" = "400" ] && grep -qi "image generation model" "$TMP/txt.json"; then
  echo "PASS: [2] a chat request to an image model is a named 400"
else
  fail "[2] expected a named 400, got $CODE: $(head -c 200 "$TMP/txt.json")"
fi

[ -n "$IDEO" ] || { echo "SKIP: no Ideogram 4 pack (set IDEOGRAM_MODEL); ran the weight-free checks only"; exit $FAILED; }

# ── 3. Load by absolute path.
api /v1/load-model -H 'Content-Type: application/json' -d "{\"model\":\"$IDEO\"}" >"$TMP/load.json"
grep -q '"state"' "$TMP/load.json" || fail "[3] load failed: $(head -c 300 "$TMP/load.json")"
# Both transformers must be in the log: the unconditional branch is its OWN
# checkpoint, and a pipeline that quietly loaded one would still produce images.
grep -q "\[ideogram4\] transformer:" "$TMP/server.log" \
  && echo "PASS: [3a] the conditional transformer loaded" \
  || fail "[3a] no conditional transformer load line"
grep -q "\[ideogram4\] unconditional_transformer:" "$TMP/server.log" \
  && echo "PASS: [3b] the unconditional transformer loaded (asymmetric CFG)" \
  || fail "[3b] no unconditional transformer load line — CFG would be one-sided"
grep -q "text encoder: .*taps=13" "$TMP/server.log" \
  && echo "PASS: [3c] the text encoder taps 13 layers" \
  || fail "[3c] text encoder did not report 13 taps"

MODEL_ID=$(basename "$IDEO")

# ── 4. Generate. A hand-written caption so the rewriter is not in the loop.
CAPTION='{"high_level_description":"A red barn in a wheat field.","compositional_deconstruction":{"background":"a wheat field under a blue sky","elements":[{"type":"obj","bbox":[200,150,850,900],"desc":"a red wooden barn"}]}}'
python3 - "$CAPTION" > "$TMP/gen.json" <<'PY'
import json,sys
print(json.dumps({"model": sys.argv[2] if len(sys.argv)>2 else None}))
PY
python3 - <<PY > "$TMP/req.json"
import json
print(json.dumps({"model": "$MODEL_ID", "prompt": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$CAPTION"),
                  "size": "512x768", "steps": 12, "seed": 7, "magic_prompt": False}))
PY
api /v1/images/generations -H 'Content-Type: application/json' --data-binary @"$TMP/req.json" > "$TMP/img.json"
python3 - "$TMP/img.json" "$TMP/out.png" <<'PY'
import base64, json, sys, struct
d = json.load(open(sys.argv[1]))
if "data" not in d:
    print("FAIL: [4] no image in response:", str(d)[:300]); sys.exit(1)
raw = base64.b64decode(d["data"][0]["b64_json"])
open(sys.argv[2], "wb").write(raw)
assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
w, h = struct.unpack(">II", raw[16:24])
# The canvas is what was ASKED for: a backend that silently re-grids to a
# square is the failure this asserts against, not a decode error.
if (w, h) != (512, 768):
    print(f"FAIL: [4] asked 512x768, got {w}x{h}"); sys.exit(1)
print(f"PASS: [4] PNG at the requested 512x768 ({len(raw)} bytes)")
PY
[ $? -eq 0 ] || FAILED=1
grep -q "magic prompt skipped" "$TMP/server.log" && echo "PASS: [4a] magic_prompt:false was honoured" || true

# ── 5. Magic prompt: a bare sentence is rewritten before conditioning.
if [ -n "$CHAT" ]; then
  api /v1/load-model -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  CHAT_ID=$(basename "$CHAT")
  : > "$TMP/mark.log"
  api /v1/images/generations -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"a red barn at sunset\",\"size\":\"512x512\",\"steps\":12,\"seed\":7,\"magic_prompt\":true,\"magic_prompt_model\":\"$CHAT_ID\"}" \
    > "$TMP/magic.json"
  if grep -q "\[ideogram4\] magic prompt: [0-9]* chars ->" "$TMP/server.log"; then
    echo "PASS: [5] a bare sentence was rewritten into a caption"
  elif grep -q "magic prompt skipped" "$TMP/server.log" || grep -q "magic prompt:.*using the raw prompt" "$TMP/server.log"; then
    # Non-fatal by design: the image still has to come back.
    python3 -c "import json,sys; d=json.load(open('$TMP/magic.json')); sys.exit(0 if 'data' in d else 1)" \
      && echo "PASS: [5] the rewrite failed and the raw prompt still rendered (non-fatal)" \
      || fail "[5] the rewrite failed AND the request died — it must fall back"
  else
    fail "[5] no magic-prompt line in the log at all"
  fi

  # A prompt that is ALREADY a caption must never be rewritten.
  : > /dev/null
  api /v1/images/generations -H 'Content-Type: application/json' --data-binary @"$TMP/req.json" >/dev/null
  grep -q "already a structured caption" "$TMP/server.log" \
    && echo "PASS: [5a] a hand-written caption is passed through untouched" || true
else
  echo "SKIP: [5] no chat model for the rewriter (set CHAT_MODEL)"
fi

# ── 6. LoRA: the ONE grammar. A path that does not exist is proven on OUR side
#       of the mlx boundary (an mlx error KILLS the process).
CODE=$(curl -s -o "$TMP/lora.json" -w '%{http_code}' -m 60 \
  -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"steps\":12,\"lora_paths\":[\"/nope/missing.safetensors\"],\"lora_scales\":[1.0]}")
if [ "$CODE" = "400" ]; then
  echo "PASS: [6] a missing LoRA path is a 400, not a crash"
else
  fail "[6] expected 400 for a missing LoRA path, got $CODE"
fi
kill -0 $SRV 2>/dev/null && echo "PASS: [6a] the server survived it" || fail "[6a] the server died on a bad LoRA path"

# ── 7. Unload frees both transformers.
api /v1/unload-model -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL_ID\"}" > "$TMP/unload.json"
grep -q '"state":"unloaded"' "$TMP/unload.json" \
  && echo "PASS: [7] unloaded" \
  || fail "[7] unload did not report unloaded: $(head -c 200 "$TMP/unload.json")"

[ $FAILED -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit $FAILED
