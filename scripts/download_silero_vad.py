"""Download the Silero VAD v5 ONNX model (~2 MB, MIT license).

Needed only when OMILOG_VAD_BACKEND=silero. Fetched from the silero-vad
GitHub repo at a pinned release tag — master moves and the v5/v4 model
interfaces differ, so don't "upgrade" the URL casually: the wrapper in
src/omilog/pipeline/silero.py targets the v5 input signature.

Idempotent — re-runs are no-ops if the file already exists.

Usage:
    .venv/bin/python scripts/download_silero_vad.py
"""

import sys
import urllib.error
import urllib.request

from omilog.config import settings

MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    "v5.1.2/src/silero_vad/data/silero_vad.onnx"
)


def main() -> int:
    target = settings.vad_silero_model
    if target.exists():
        print(f"✓ silero VAD model already present: {target}")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    print(f"  ↓ {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL) as response, tmp.open("wb") as out:
        out.write(response.read())
    tmp.rename(target)
    print(f"✓ silero VAD model ready: {target} ({target.stat().st_size // 1024} KB)")
    print()
    print("Enable with OMILOG_VAD_BACKEND=silero (in .env or /config), then restart.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as e:
        print(f"\nnetwork error: {e}", file=sys.stderr)
        print("Check connectivity to raw.githubusercontent.com", file=sys.stderr)
        raise SystemExit(1)
