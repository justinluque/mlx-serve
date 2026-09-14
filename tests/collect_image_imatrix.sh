#!/usr/bin/env bash
# Collect an imatrix for an image pack by generating from the calibration corpus.
#
#   tests/collect_image_imatrix.sh <model_dir> <out.safetensors> [port]
#
# The engine accumulates per-input-channel E[x^2] for every projection of the
# transformer and text encoder while it generates (MLX_SERVE_IMATRIX) and
# rewrites <out> after each image, so an interrupted run keeps what it measured.
# Collect through the highest-precision pack that fits: the statistic is an
# activation moment, far steadier under weight noise than the weights, but a
# 4-bit pack is still an approximation of the source.
#
# Env: STEPS (Krea 8, Z-Image-Turbo 8, klein 4, a dir named *base* 30), GUIDANCE (base default
#      4.0), SIZES (csv WxH), SEEDS_PER_PROMPT (1), LIMIT (stop after N images),
#      PROMPTS (tests/fixtures/image_calibration_prompts.txt).
set -euo pipefail

MODEL=${1:?usage: collect_image_imatrix.sh <model_dir> <out.safetensors> [port]}
OUT=${2:?output .safetensors path}
PORT=${3:-11420}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="$ROOT/zig-out/bin/mlx-serve"
PROMPTS=${PROMPTS:-$ROOT/tests/fixtures/image_calibration_prompts.txt}
SIZES=${SIZES:-1024x1024,1344x768,768x1344,896x1152}
SEEDS_PER_PROMPT=${SEEDS_PER_PROMPT:-1}
[[ -x "$BIN" ]] || { echo "build first: zig build -Doptimize=ReleaseFast"; exit 1; }
[[ -d "$MODEL" ]] || { echo "no model dir: $MODEL"; exit 1; }
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac

model_type=$(python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1] + "/config.json")).get("model_type", ""))
except Exception:
  try: print("zimage" if json.load(open(sys.argv[1] + "/model_index.json")).get("_class_name") == "ZImagePipeline" else "")
  except Exception: print("")' "$MODEL")
if [[ "$model_type" == krea* ]]; then
  STEPS=${STEPS:-8}
elif [[ "$model_type" == zimage ]]; then
  # z_image.dirLooksTurbo: "turbo" anywhere in the path picks the sampler.
  [[ "$(echo "$MODEL" | tr A-Z a-z)" == *turbo* ]] && STEPS=${STEPS:-8} || STEPS=${STEPS:-50}
elif [[ "$(basename "$MODEL" | tr A-Z a-z)" == *base* ]]; then
  STEPS=${STEPS:-30}
  GUIDANCE=${GUIDANCE:-4.0}
else
  STEPS=${STEPS:-4}
fi
GUIDANCE=${GUIDANCE:-}

LOG="${OUT%.safetensors}.server.log"
MLX_SERVE_IMATRIX="$OUT" "$BIN" --serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" --log-file "$LOG" >/dev/null 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "server exited during boot (see $LOG)"; exit 1; }
  sleep 1
done

IFS=, read -ra SIZE_LIST <<<"$SIZES"
n=0
while IFS= read -r prompt; do
  [[ -z "$prompt" ]] && continue
  for ((s = 0; s < SEEDS_PER_PROMPT; s++)); do
    size=${SIZE_LIST[$((n % ${#SIZE_LIST[@]}))]}
    body=$(python3 -c 'import json, sys
b = {"prompt": sys.argv[1], "size": sys.argv[2], "seed": int(sys.argv[3]), "steps": int(sys.argv[4])}
if sys.argv[5]: b["guidance_scale"] = float(sys.argv[5])
print(json.dumps(b))' "$prompt" "$size" "$((1000 + n))" "$STEPS" "$GUIDANCE")
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
      -H 'Content-Type: application/json' -d "$body")
    [[ "$code" == 200 ]] || { echo "generation $n failed: HTTP $code (see $LOG)"; exit 1; }
    n=$((n + 1))
    echo "[$n] $size steps=$STEPS ${GUIDANCE:+guidance=$GUIDANCE }$prompt"
    if [[ -n "${LIMIT:-}" && $n -ge $LIMIT ]]; then break 2; fi
  done
done <"$PROMPTS"

grep -q 'imatrix armed' "$LOG" || { echo "FAIL: the engine never armed the collector (see $LOG)"; exit 1; }
grep '\[imatrix\] .* projections ->' "$LOG" | tail -1
echo "$n images -> $OUT"
