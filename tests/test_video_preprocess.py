"""Control-video preprocessing matches upstream ``decode_video_by_frame`` + ``video_preprocess`` (issue #194)."""

import shutil
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ltx_pipelines_mlx.ic_lora import _load_mask_video
from ltx_pipelines_mlx.utils.media_io import decode_video_by_frame, resize_and_center_crop, video_preprocess

FIXTURES = Path(__file__).parent / "fixtures"
CLIP = FIXTURES / "video_preprocess_160x90_9f.mp4"
GOLDEN = FIXTURES / "video_preprocess_golden.npz"
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed"
)


def _f32(x: mx.array) -> np.ndarray:
    return np.asarray(x.astype(mx.float32))


@needs_ffmpeg
def test_decode_matches_upstream_pyav() -> None:
    """Native size, every frame, same RGB as PyAV's ``to_rgb()``."""
    golden = np.load(GOLDEN)
    frames = list(decode_video_by_frame(CLIP))
    assert len(frames) == int(golden["decoded_count"]) == 9
    assert frames[0].shape == (90, 160, 3) and frames[0].dtype == np.uint8
    np.testing.assert_array_equal(np.stack([frames[0], frames[-1]]), golden["decoded_first_last"])


@needs_ffmpeg
@pytest.mark.parametrize("case", [0, 1, 2])
def test_video_preprocess_matches_upstream(case: int) -> None:
    """Targets with an aspect ratio different from the 16:9 source: center crop, not stretch."""
    golden = np.load(GOLDEN)
    height, width, start, cap = (int(v) for v in golden[f"case_{case}"])
    out = video_preprocess(decode_video_by_frame(CLIP, starting_frame=start, frame_cap=cap), height, width)
    expected = golden[f"out_{case}"].astype(np.float32)
    assert out.dtype == mx.bfloat16
    assert out.shape == expected.shape == (1, 3, cap, height, width)
    # bfloat16 output vs a float16 golden: both round values in [-1, 1] at ~4e-3.
    np.testing.assert_allclose(_f32(out), expected, atol=1e-2, rtol=0)


@needs_ffmpeg
def test_decode_frame_cap_and_start() -> None:
    all_frames = list(decode_video_by_frame(CLIP))
    some = list(decode_video_by_frame(CLIP, starting_frame=3, frame_cap=4))
    assert len(some) == 4
    for a, b in zip(some, all_frames[3:7], strict=True):
        np.testing.assert_array_equal(a, b)


@needs_ffmpeg
def test_decode_ignores_rotation_metadata(tmp_path: Path) -> None:
    """PyAV does not apply the display matrix, so frames keep the coded size."""
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(CLIP), "-c", "copy", str(rotated)],
        check=True,
    )
    frames = list(decode_video_by_frame(rotated, frame_cap=1))
    assert frames[0].shape == (90, 160, 3)


@needs_ffmpeg
def test_decode_raises_on_unreadable_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError):
        list(decode_video_by_frame(bad))


def test_video_preprocess_is_resize_and_center_crop_per_frame() -> None:
    rng = np.random.default_rng(194)
    frames = [rng.integers(0, 256, (30, 50, 3), dtype=np.uint8) for _ in range(3)]
    out = video_preprocess(iter(frames), 20, 20)
    assert out.shape == (1, 3, 3, 20, 20)
    for i, frame in enumerate(frames):
        expected = resize_and_center_crop(frame, 20, 20) / np.float32(127.5) - np.float32(1.0)
        expected_bf16 = _f32(mx.array(expected.transpose(2, 0, 1)).astype(mx.bfloat16))
        np.testing.assert_array_equal(_f32(out[0, :, i]), expected_bf16)


def test_video_preprocess_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty frame generator"):
        video_preprocess(iter([]), 8, 8)


@needs_ffmpeg
def test_mask_video_spans_zero_to_one(tmp_path: Path) -> None:
    """Black → 0 and white → 1. The old loader returned [0, 1] pixels but remapped them as [-1, 1],
    so a black mask came out as 0.5."""
    path = tmp_path / "mask.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", "color=c=black:s=64x32:r=24,drawbox=x=32:y=0:w=32:h=32:color=white:t=fill",
            "-frames:v", "9", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path),
        ],
        check=True,
    )  # fmt: skip
    mask = _load_mask_video(str(path), 32, 64, 9)
    assert mask.shape == (1, 1, 9, 32, 64)
    values = _f32(mask)
    assert values[..., :24].max() < 0.02  # black half
    assert values[..., 40:].min() > 0.98  # white half
