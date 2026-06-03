"""Download sherpa-onnx diarization models.

All-local diarization needs two ONNX files:
  - sherpa-onnx-pyannote-segmentation-3-0 (~12 MB) — speech-vs-silence + speaker
    boundaries. Same model as Phase 4's pyannote, just ONNX-converted.
  - nemo_en_titanet_small (~28 MB) — speaker embedding model for clustering.
    Trained on English speakers but speaker embeddings are language-agnostic
    in practice; works fine for French.

Both are hosted on the sherpa-onnx GitHub releases (Apache 2 license, no
account required, no license click-through).

Idempotent — re-runs are no-ops if the files already exist.

Usage:
    .venv/bin/python scripts/download_diarization_models.py
"""

import bz2
import sys
import tarfile
import urllib.request
from pathlib import Path

from omilog.config import settings


SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_DIR_NAME = "sherpa-onnx-pyannote-segmentation-3-0"

EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "nemo_en_titanet_small.onnx"
)


def _download(url: str, dst: Path) -> None:
    print(f"  ↓ {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        total = int(response.headers.get("content-length") or 0)
        chunk = 1 << 16
        downloaded = 0
        last_pct = -1
        while True:
            data = response.read(chunk)
            if not data:
                break
            out.write(data)
            downloaded += len(data)
            if total:
                pct = int(downloaded * 100 / total)
                if pct != last_pct and pct % 10 == 0:
                    print(f"    {pct}% ({downloaded // 1024} KB / {total // 1024} KB)")
                    last_pct = pct
    tmp.rename(dst)


def ensure_segmentation() -> Path:
    target_dir = settings.diarization_models_dir / SEGMENTATION_DIR_NAME
    target_onnx = target_dir / "model.onnx"
    if target_onnx.exists():
        print(f"✓ segmentation already present: {target_onnx}")
        return target_onnx

    archive = settings.diarization_models_dir / "segmentation.tar.bz2"
    _download(SEGMENTATION_URL, archive)
    print(f"  ⇲ extracting → {settings.diarization_models_dir}")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(settings.diarization_models_dir)
    archive.unlink()
    if not target_onnx.exists():
        raise RuntimeError(
            f"extracted but model.onnx not found at {target_onnx}; "
            f"archive layout may have changed"
        )
    print(f"✓ segmentation ready: {target_onnx}")
    return target_onnx


def ensure_embedding() -> Path:
    target = settings.diarization_models_dir / "nemo_en_titanet_small.onnx"
    if target.exists():
        print(f"✓ embedding already present: {target}")
        return target
    _download(EMBEDDING_URL, target)
    print(f"✓ embedding ready: {target}")
    return target


def main() -> int:
    settings.diarization_models_dir.mkdir(parents=True, exist_ok=True)
    print(f"models dir: {settings.diarization_models_dir.resolve()}")
    print()

    print("→ segmentation model")
    seg = ensure_segmentation()
    print()

    print("→ embedding model")
    emb = ensure_embedding()
    print()

    print("Done. Verify the paths in .env match:")
    print(f"  OMILOG_DIARIZATION_SEGMENTATION_MODEL={seg}")
    print(f"  OMILOG_DIARIZATION_EMBEDDING_MODEL={emb}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as e:
        print(f"\nnetwork error: {e}", file=sys.stderr)
        print("Check connectivity to github.com / objects.githubusercontent.com", file=sys.stderr)
        raise SystemExit(1)
