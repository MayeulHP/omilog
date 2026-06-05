"""Download sherpa-onnx diarization models.

All-local diarization needs two ONNX files:
  1. **Segmentation** — pyannote-segmentation-3-0, ONNX-converted. Used for
     voice-activity detection and turn boundaries. One choice, ~12 MB.
  2. **Embedding** — produces a per-speaker vector from a chunk of audio.
     The Speaker-clustering / cross-conversation linking layer uses cosine
     similarity over these vectors. Multiple options exist; downloading
     more than one is cheap (~10-30 MB each) so by default we fetch both
     candidates and let the user pick via ``OMILOG_DIARIZATION_EMBEDDING_MODEL``.

All hosted on the sherpa-onnx GitHub releases (Apache 2 license, no account
required).

Idempotent — re-runs are no-ops if the files already exist.

Usage:
    .venv/bin/python scripts/download_diarization_models.py
    .venv/bin/python scripts/download_diarization_models.py --only nemo-titanet-small
    .venv/bin/python scripts/download_diarization_models.py --list
"""

import argparse
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


# Catalogue of embedding models. Keep keys short and dasherized — they're
# what users type after --only. Each entry: (display_filename, full URL,
# short description shown by --list).
EMBEDDING_MODELS: dict[str, tuple[str, str, str]] = {
    "nemo-titanet-small": (
        "nemo_en_titanet_small.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/nemo_en_titanet_small.onnx",
        "~28 MB. NVIDIA NeMo TitaNet, trained on English telephony. The "
        "original omilog default. Works on French but tends to cluster "
        "same-gender French speakers together; less stable on short "
        "(<3 s) utterances.",
    ),
    "3dspeaker-eres2net-en": (
        "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/"
        "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        "~10 MB. 3D-Speaker ERes2Net, VoxCeleb-trained. Multi-scale "
        "fusion architecture produces more stable embeddings on short "
        "utterances; VoxCeleb training pool is more speaker-diverse than "
        "TitaNet's English telephony bias, so usually better for non-"
        "English voices. Recommended first attempt if TitaNet is "
        "over-segmenting your data.",
    ),
}

# When the user runs without --only, fetch this set. Both candidates are
# cheap on disk so it's worth having them ready for trial-and-error.
DEFAULT_EMBEDDINGS: tuple[str, ...] = (
    "nemo-titanet-small",
    "3dspeaker-eres2net-en",
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


def ensure_embedding(key: str) -> Path:
    if key not in EMBEDDING_MODELS:
        raise KeyError(
            f"unknown embedding model key: {key!r}. "
            f"Run with --list to see options."
        )
    filename, url, _ = EMBEDDING_MODELS[key]
    target = settings.diarization_models_dir / filename
    if target.exists():
        print(f"✓ {key} already present: {target}")
        return target
    _download(url, target)
    print(f"✓ {key} ready: {target}")
    return target


def _list_options() -> None:
    print("Available embedding models:")
    for key, (filename, _, desc) in EMBEDDING_MODELS.items():
        marker = " (default)" if key in DEFAULT_EMBEDDINGS else ""
        print(f"\n  {key}{marker}")
        print(f"    file: models/{filename}")
        print(f"    {desc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        metavar="KEY",
        help=(
            "Download only the listed embedding model(s) instead of every "
            "default. Pass multiple times for several. "
            "Use --list to see keys."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List embedding model keys + descriptions and exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        _list_options()
        return 0

    settings.diarization_models_dir.mkdir(parents=True, exist_ok=True)
    print(f"models dir: {settings.diarization_models_dir.resolve()}")
    print()

    print("→ segmentation model")
    seg = ensure_segmentation()
    print()

    targets = tuple(args.only) if args.only else DEFAULT_EMBEDDINGS
    for key in targets:
        if key not in EMBEDDING_MODELS:
            print(
                f"\nunknown model {key!r} — run with --list to see options",
                file=sys.stderr,
            )
            return 2

    embeddings: dict[str, Path] = {}
    for key in targets:
        print(f"→ embedding model: {key}")
        embeddings[key] = ensure_embedding(key)
        print()

    print("Done. Configure omilog by pointing the env var at the file you want:")
    print(f"  OMILOG_DIARIZATION_SEGMENTATION_MODEL={seg}")
    print()
    print("  # pick ONE of these embedding paths:")
    for key, path in embeddings.items():
        marker = "  ← current default" if key == "nemo-titanet-small" else ""
        print(f"  OMILOG_DIARIZATION_EMBEDDING_MODEL={path}{marker}")
    print()
    print(
        "(Restart the server after changing the embedding model — the "
        "diarizer is process-cached.)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as e:
        print(f"\nnetwork error: {e}", file=sys.stderr)
        print(
            "Check connectivity to github.com / objects.githubusercontent.com",
            file=sys.stderr,
        )
        raise SystemExit(1)
