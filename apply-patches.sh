#!/bin/bash
# Apply patches/*.patch to the installed vllm-omni 0.32.0 and vllm 0.32.0. Safe to run twice.
set -euo pipefail
PATCHES=${PATCHES:-$(cd "$(dirname "$0")" && pwd)/patches}
PYTHON=${PYTHON:-python3}
SITE=$("$PYTHON" -c 'import pathlib, vllm_omni; print(pathlib.Path(vllm_omni.__file__).parents[1])')
"$PYTHON" - "$SITE" <<'PY'
import pathlib, sys, vllm, vllm_omni
site = pathlib.Path(sys.argv[1])
if pathlib.Path(vllm.__file__).parents[1] != site:
    sys.exit('vllm and vllm_omni must be installed in the same site-packages')
for name, version in (('vllm', vllm.__version__), ('vllm_omni', vllm_omni.__version__)):
    if not version.startswith('0.32.0'):
        print(f'warning: patches target {name} 0.32.0, found {version}', file=sys.stderr)
PY
shopt -s nullglob
for patch in "$PATCHES"/*.patch; do
  if patch -d "$SITE" -p1 -R --dry-run -s -f < "$patch" > /dev/null 2>&1; then
    echo "already applied $(basename "$patch")"
    continue
  fi
  patch -d "$SITE" -p1 -N -s --no-backup-if-mismatch < "$patch"
  echo "applied $(basename "$patch")"
done
