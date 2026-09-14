#!/usr/bin/env bash
# Z-Image / Z-Image-Turbo image gen: headless boot -> load by absolute path ->
# generate -> assert a valid PNG of the requested size -> unload. Proves the
# image-backend seam routes `zimage*` to the Z-Image engine end to end.
#
# No numeric oracle exists for this backend (see docs/reference.md "Z-Image" +
# CLAUDE.md) — this script is the actual acceptance bar: a real image out.
#
# Skips gracefully when no Z-Image model is present. The dir must be the
# diffusers layout (model_index.json + transformer/ + text_encoder/ + vae/ +
# tokenizer/), produced by tests/convert_zimage_weights.py.
#
# Usage: ZIMAGE_MODEL=<dir> ./tests/test_zimage.sh [port]
set -uo pipefail
PORT="${1:-11398}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

ZIMG="${ZIMAGE_MODEL:-$(ls -d ~/.mlx-serve/models/*/Z-Image-Turbo* 2>/dev/null | head -1)}"
[ -n "$ZIMG" ] || { echo "SKIP: no Z-Image model (set ZIMAGE_MODEL to an assembled dir)"; exit 0; }
[ -f "$ZIMG/model_index.json" ] || { echo "SKIP: $ZIMG has no model_index.json (not a diffusers-shaped Z-Image dir)"; exit 0; }

HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_zimage_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_zimage_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 1800 "http://127.0.0.1:$PORT$1" "${@:2}"; }
ZIMG_ID="$(basename "$ZIMG")"

# 1. Load Z-Image by absolute path -> ready with "image" capability.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ZIMG\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ZIMG_ID' and x['state']=='ready' and 'image' in x.get('capabilities',[])]
assert m, 'Z-Image not ready with image cap: '+json.dumps(d)
print('PASS: load-model by path -> Z-Image ready, capabilities', m[0]['capabilities'])
"
[ $? -eq 0 ] || exit 1

# 2. Generate at 512x512 -> PNG of that size (Turbo defaults: 8 steps, no CFG).
api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ZIMG_ID\",\"prompt\":\"a red fox in the snow\",\"size\":\"512x512\"}" \
  -o /tmp/test_zimage_img.json -w ''
python3 - /tmp/test_zimage_img.json <<'PY'
import sys,json,base64,struct
b=base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
w,h=struct.unpack(">II", b[16:24])
assert (w,h)==(512,512), f"expected 512x512, got {w}x{h}"
print(f"PASS: generated {w}x{h} PNG ({len(b)} bytes)")
PY
[ $? -eq 0 ] || exit 1

# 3. Unload.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ZIMG_ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ZIMG_ID']
assert not m or m[0]['state']!='ready', 'Z-Image still ready after unload'
print('PASS: unload')
"
[ $? -eq 0 ] || exit 1

echo "PASS: test_zimage.sh"
