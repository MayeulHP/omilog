"""Verify the Ogg-Opus muxer produces structurally valid files.

Two layers:
  - parse_pages re-implements Ogg framing top-down so we can assert structure
    independently of the writer (otherwise we'd be checking a thing against
    itself).
  - If ffmpeg is on PATH, we also shell out to it as a third-party oracle.
"""

import io
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from omilog.audio.ogg_opus import (
    OggOpusWriter,
    _segment_table,
    ogg_crc32,
)


# ──────────────────────────────────────────────────────────────────────────────
# Independent Ogg page parser (for assertions only — kept simple, not a lib)
# ──────────────────────────────────────────────────────────────────────────────

def parse_pages(data: bytes):
    """Yield dicts describing each Ogg page in `data`."""
    pos = 0
    while pos < len(data):
        if data[pos : pos + 4] != b"OggS":
            raise AssertionError(f"missing OggS magic at offset {pos}")
        version = data[pos + 4]
        flags = data[pos + 5]
        granule = struct.unpack_from("<q", data, pos + 6)[0]
        serial = struct.unpack_from("<I", data, pos + 14)[0]
        seq = struct.unpack_from("<I", data, pos + 18)[0]
        crc_stored = struct.unpack_from("<I", data, pos + 22)[0]
        n_segs = data[pos + 26]
        seg_table = data[pos + 27 : pos + 27 + n_segs]
        payload_len = sum(seg_table)
        payload_start = pos + 27 + n_segs
        payload = data[payload_start : payload_start + payload_len]
        end = payload_start + payload_len

        # Verify CRC by zeroing those 4 bytes and recomputing.
        page = bytearray(data[pos:end])
        struct.pack_into("<I", page, 22, 0)
        crc_computed = ogg_crc32(bytes(page))

        yield {
            "version": version,
            "flags": flags,
            "granule": granule,
            "serial": serial,
            "seq": seq,
            "crc_stored": crc_stored,
            "crc_computed": crc_computed,
            "segments": list(seg_table),
            "payload": bytes(payload),
        }
        pos = end


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests
# ──────────────────────────────────────────────────────────────────────────────

def test_segment_table_small():
    assert _segment_table(0) == bytes([0])
    assert _segment_table(1) == bytes([1])
    assert _segment_table(100) == bytes([100])
    assert _segment_table(254) == bytes([254])


def test_segment_table_boundary():
    # Exactly 255 needs [255, 0] to mark packet end with a sub-255 segment.
    assert _segment_table(255) == bytes([255, 0])
    assert _segment_table(256) == bytes([255, 1])
    assert _segment_table(510) == bytes([255, 255, 0])
    assert _segment_table(511) == bytes([255, 255, 1])


def test_headers_are_first_two_pages():
    buf = io.BytesIO()
    OggOpusWriter(buf, sample_rate=16000, channels=1).close()
    pages = list(parse_pages(buf.getvalue()))
    assert len(pages) == 3  # id, comment, eos

    id_head, comment_head, eos = pages
    assert id_head["payload"].startswith(b"OpusHead")
    assert id_head["flags"] & 0x02, "id header must have BOS flag"
    assert id_head["seq"] == 0

    assert comment_head["payload"].startswith(b"OpusTags")
    assert comment_head["seq"] == 1

    assert eos["flags"] & 0x04, "last page must have EOS flag"
    assert eos["payload"] == b""


def test_opushead_structure():
    buf = io.BytesIO()
    OggOpusWriter(buf, sample_rate=16000, channels=1).close()
    pages = list(parse_pages(buf.getvalue()))
    head = pages[0]["payload"]
    assert head[:8] == b"OpusHead"
    assert head[8] == 1               # version
    assert head[9] == 1               # channels
    pre_skip = struct.unpack_from("<H", head, 10)[0]
    rate = struct.unpack_from("<I", head, 12)[0]
    assert pre_skip == 0
    assert rate == 16000
    assert head[18] == 0              # channel mapping family 0


def test_packets_round_trip():
    buf = io.BytesIO()
    w = OggOpusWriter(buf, sample_rate=16000, channels=1)
    # Use packets of varying sizes incl. boundary cases at 255/256.
    packets = [b"\xab" * n for n in (40, 80, 254, 255, 256, 500)]
    for p in packets:
        w.write_packet(p)
    w.close()

    pages = list(parse_pages(buf.getvalue()))
    # 2 header pages + len(packets) data pages + 1 EOS page
    assert len(pages) == 2 + len(packets) + 1

    data_pages = pages[2 : 2 + len(packets)]
    for sent, got in zip(packets, data_pages):
        assert got["payload"] == sent
        assert got["crc_stored"] == got["crc_computed"]

    # Granule grows monotonically by frame_duration*48 samples each packet.
    granules = [p["granule"] for p in data_pages]
    assert granules == sorted(granules)
    # 20 ms @ 48 kHz = 960 samples per increment.
    assert granules[0] == 960
    assert granules[-1] == 960 * len(packets)


def test_empty_packets_ignored():
    buf = io.BytesIO()
    w = OggOpusWriter(buf, sample_rate=16000, channels=1)
    w.write_packet(b"")
    w.write_packet(b"\xff" * 50)
    w.close()
    assert w.packets_written == 1


def test_all_crcs_valid():
    buf = io.BytesIO()
    w = OggOpusWriter(buf, sample_rate=16000, channels=1)
    for n in (10, 100, 300):
        w.write_packet(b"\xcd" * n)
    w.close()
    for page in parse_pages(buf.getvalue()):
        assert page["crc_stored"] == page["crc_computed"], (
            f"CRC mismatch on page seq={page['seq']}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Integration: third-party validation via ffprobe (skipped if not installed)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_ffprobe_recognizes_file(tmp_path: Path):
    """We can produce a structurally valid Ogg-Opus, but ffprobe will reject
    the *audio* content because our 'packets' are random bytes, not real Opus.
    What we assert: ffprobe identifies the container as ogg/opus."""
    out = tmp_path / "fake.opus"
    with out.open("wb") as f:
        w = OggOpusWriter(f, sample_rate=16000, channels=1)
        # 50 fake "packets" of ~100 bytes each
        for _ in range(50):
            w.write_packet(b"\x00" * 100)
        w.close()

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", str(out)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    # ffprobe may warn about invalid packets, but container detection should
    # at least produce "ogg" in the format line if it parses pages.
    combined = result.stdout + result.stderr
    assert "ogg" in combined.lower(), f"ffprobe didn't recognize Ogg: {combined!r}"
