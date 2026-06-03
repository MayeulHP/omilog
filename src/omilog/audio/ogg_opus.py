"""Stream raw Opus packets into an Ogg container.

References:
  RFC 3533 — The Ogg Encapsulation Format Version 0
  RFC 7845 — Ogg Encapsulation for the Opus Audio Codec
  https://wiki.xiph.org/OggOpus

What this does
--------------
Wyoming sends one Opus packet per WS binary frame; we receive raw packets but
need a container so that ffmpeg / whisper.cpp can find packet boundaries. Doing
the muxing while we receive (rather than re-encoding at session close) means
we don't have to keep a sidecar manifest, and a crash mid-session still leaves
a playable prefix.

Assumptions
-----------
- One Wyoming WS binary frame == one Opus packet. (Holds in the captures we've
  seen; mitmproxy can prove or disprove this.)
- Frame duration is 20 ms unless overridden. Most speech-tuned Opus streams use
  20 ms frames. If the actual frame is 10 / 40 ms, the resulting file still
  plays — only the per-page granule position (used for seek) is off, by a
  factor of (actual / 20). We document this rather than decoding each packet,
  because decoding would require libopus and the cost of being wrong is "seek
  doesn't land exactly," not data loss.
- Granule position is in PCM samples at 48 kHz (RFC 7845 §4), not at the
  capture sample rate. 20 ms == 960 samples at 48 kHz.

Limits
------
- Up to 255 lacing segments per page → max ~65,025 bytes per packet, which is
  far above what speech codecs produce (~hundreds of bytes). No multi-page
  packet handling implemented.
- One packet per page. Wastes ~30 bytes/page (~1.5 KB/s at 50 fps), acceptable
  for our scale; could batch later if it matters.
"""

from __future__ import annotations

import struct
from typing import BinaryIO


# ──────────────────────────────────────────────────────────────────────────────
# Ogg CRC-32 (RFC 3533 §6) — polynomial 0x04C11DB7, init 0, no final XOR.
# Different from the more common CRC-32/IEEE used in zip/png; can't use zlib.
# ──────────────────────────────────────────────────────────────────────────────

def _make_crc_table() -> list[int]:
    table = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) if (r & 0x80000000) else (r << 1)
            r &= 0xFFFFFFFF
        table.append(r)
    return table


_CRC_TABLE = _make_crc_table()


def ogg_crc32(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((crc << 8) ^ _CRC_TABLE[((crc >> 24) ^ b) & 0xFF]) & 0xFFFFFFFF
    return crc


# ──────────────────────────────────────────────────────────────────────────────
# Header flag bits (RFC 3533 §6)
# ──────────────────────────────────────────────────────────────────────────────

_FLAG_CONTINUED = 0x01  # this page is a continuation of a previous packet
_FLAG_FIRST = 0x02      # first page of logical bitstream (BOS)
_FLAG_LAST = 0x04       # last page of logical bitstream (EOS)


def _segment_table(payload_len: int) -> bytes:
    """Build the lacing table for a single packet that ends on this page.

    The packet is split into 255-byte segments; the final segment must be <255
    to mark the packet boundary (even a 0-byte terminator when the packet is an
    exact multiple of 255).
    """
    if payload_len > 255 * 255:
        raise ValueError(
            f"packet too large for single-page muxer: {payload_len} bytes"
        )
    full, remainder = divmod(payload_len, 255)
    segments = bytes([255] * full + [remainder])
    return segments


def _build_page(
    payload: bytes,
    *,
    flags: int,
    granule: int,
    serial: int,
    seq: int,
) -> bytes:
    seg_table = _segment_table(len(payload))
    if len(seg_table) > 255:
        raise ValueError(
            f"segment table overflow: {len(seg_table)} > 255 segments"
        )
    header = bytearray()
    header += b"OggS"                          # capture pattern
    header += b"\x00"                          # stream structure version
    header += struct.pack("<B", flags)
    header += struct.pack("<q", granule)       # granule position (signed 64)
    header += struct.pack("<I", serial)        # bitstream serial number
    header += struct.pack("<I", seq)           # page sequence number
    header += struct.pack("<I", 0)             # CRC placeholder (filled below)
    header += struct.pack("<B", len(seg_table))
    header += seg_table

    page = bytes(header) + payload
    crc = ogg_crc32(page)
    # Patch CRC into bytes 22..26
    return page[:22] + struct.pack("<I", crc) + page[26:]


# ──────────────────────────────────────────────────────────────────────────────
# Public writer
# ──────────────────────────────────────────────────────────────────────────────

OPUS_INTERNAL_SAMPLE_RATE = 48000  # RFC 7845: granule is always at 48 kHz


class OggOpusWriter:
    """Stream Opus packets into an Ogg file.

    Usage:
        with open("session.opus", "wb") as f:
            w = OggOpusWriter(f, sample_rate=16000, channels=1)
            for pkt in opus_packets:
                w.write_packet(pkt)
            w.close()
    """

    def __init__(
        self,
        out: BinaryIO,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration_ms: float = 20.0,
        serial: int = 0xCAFE0001,
        vendor: bytes = b"omilog",
    ) -> None:
        self._out = out
        self._sample_rate = sample_rate
        self._channels = channels
        self._serial = serial
        self._vendor = vendor
        self._seq = 0
        self._granule = 0
        self._closed = False
        # Samples per packet at Opus's internal 48 kHz, used for granule pos.
        self._granule_increment = int(
            OPUS_INTERNAL_SAMPLE_RATE * frame_duration_ms / 1000
        )
        self._packets_written = 0
        self._write_id_header()
        self._write_comment_header()

    # ── headers ──────────────────────────────────────────────────────────────

    def _write_id_header(self) -> None:
        # RFC 7845 §5.1: 19 bytes for mono Channel Mapping Family 0.
        head = bytearray()
        head += b"OpusHead"
        head += b"\x01"                              # version 1
        head += struct.pack("<B", self._channels)
        head += struct.pack("<H", 0)                 # pre-skip
        head += struct.pack("<I", self._sample_rate) # input sample rate
        head += struct.pack("<h", 0)                 # output gain (Q7.8 dB)
        head += b"\x00"                              # Channel Mapping Family 0
        self._emit_page(bytes(head), flags=_FLAG_FIRST, granule=0)

    def _write_comment_header(self) -> None:
        # RFC 7845 §5.2: OpusTags + Vorbis-style comment header.
        body = bytearray()
        body += b"OpusTags"
        body += struct.pack("<I", len(self._vendor))
        body += self._vendor
        body += struct.pack("<I", 0)                 # zero user comments
        self._emit_page(bytes(body), flags=0, granule=0)

    # ── data ────────────────────────────────────────────────────────────────

    def write_packet(self, packet: bytes) -> None:
        if self._closed:
            raise ValueError("writer is closed")
        if not packet:
            # Empty packets aren't valid Opus; ignore silently to be defensive
            # against WS keepalive frames.
            return
        self._granule += self._granule_increment
        self._emit_page(packet, flags=0, granule=self._granule)
        self._packets_written += 1

    # ── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        # End-of-stream marker: empty page with EOS flag.
        self._emit_page(b"", flags=_FLAG_LAST, granule=self._granule)
        self._closed = True

    @property
    def packets_written(self) -> int:
        return self._packets_written

    @property
    def granule_position(self) -> int:
        return self._granule

    # ── internals ───────────────────────────────────────────────────────────

    def _emit_page(self, payload: bytes, *, flags: int, granule: int) -> None:
        page = _build_page(
            payload,
            flags=flags,
            granule=granule,
            serial=self._serial,
            seq=self._seq,
        )
        self._out.write(page)
        self._seq += 1
