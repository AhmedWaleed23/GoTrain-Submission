"""Minimal PNG encoder (zlib + struct only) and mask colouring for the dashboard."""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
    ],
    dtype=np.uint8,
)


def encode_png(rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def png_data_uri(rgb: np.ndarray, scale: int = 1) -> str:
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return "data:image/png;base64," + base64.b64encode(encode_png(rgb)).decode("ascii")


def colour_mask(mask: np.ndarray) -> np.ndarray:
    idx = np.clip(np.asarray(mask, dtype=np.int64), 0, len(PALETTE) - 1)
    return PALETTE[idx]
