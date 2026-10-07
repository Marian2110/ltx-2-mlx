"""Image-conditioning preprocessing matches upstream ``ltx_pipelines.utils.media_io`` (issue #188)."""

import logging
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image, ImageCms

from ltx_pipelines_mlx.utils.media_io import (
    decode_image,
    encode_single_frame,
    load_image_and_preprocess,
    preprocess,
    resize_and_center_crop,
)

GOLDEN = Path(__file__).parent / "fixtures" / "image_preprocess_resize_golden.npz"
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed"
)


def _gradient(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) % 256], axis=-1).astype(np.uint8)


def test_resize_matches_upstream_torch() -> None:
    """Bilinear on floats (``align_corners=False``, no antialias) + center crop, against upstream outputs."""
    golden = np.load(GOLDEN)
    n_cases = len([k for k in golden.files if k.startswith("src_")])
    assert n_cases == 4
    for i in range(n_cases):
        src, expected = golden[f"src_{i}"], golden[f"out_{i}"]
        out = resize_and_center_crop(src, expected.shape[0], expected.shape[1])
        assert out.dtype == np.float32
        assert out.shape == expected.shape
        # float32 rounding of the separable form vs torch's 2D kernel: ~1e-6 relative, far below one uint8 step.
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=0)


def test_resize_keeps_float_values() -> None:
    """The resized frame is not re-quantized to uint8 (upstream normalizes the float result)."""
    out = resize_and_center_crop(_gradient(37, 53), 24, 32)
    assert np.any(out != np.round(out))


def test_resize_accepts_pil() -> None:
    arr = _gradient(30, 40)
    np.testing.assert_array_equal(
        resize_and_center_crop(Image.fromarray(arr), 20, 20), resize_and_center_crop(arr, 20, 20)
    )


def test_load_image_normalizes_like_upstream(tmp_path: Path) -> None:
    """``x / 127.5 - 1`` on the float resize, then bfloat16."""
    arr = _gradient(48, 64)
    path = tmp_path / "img.png"
    Image.fromarray(arr).save(path)
    tensor = load_image_and_preprocess(path, 32, 48, crf=0)
    assert tensor.shape == (1, 3, 32, 48)
    assert tensor.dtype == mx.bfloat16
    expected = resize_and_center_crop(arr, 32, 48) / np.float32(127.5) - np.float32(1.0)
    expected_bf16 = np.asarray(mx.array(expected.transpose(2, 0, 1)[None]).astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_array_equal(np.asarray(tensor.astype(mx.float32)), expected_bf16)


@pytest.mark.parametrize(("orientation", "expected_shape"), [(1, (20, 30, 3)), (3, (20, 30, 3)), (6, (30, 20, 3))])
def test_decode_image_applies_exif_orientation(tmp_path: Path, orientation: int, expected_shape: tuple) -> None:
    arr = _gradient(20, 30)
    exif = Image.Exif()
    exif[0x0112] = orientation
    path = tmp_path / "img.jpg"
    Image.fromarray(arr).save(path, exif=exif, quality=100)
    decoded = decode_image(str(path))
    assert decoded.shape == expected_shape
    reference = np.asarray(Image.open(path).convert("RGB"))
    rotation = {3: 180, 6: 270}.get(orientation)
    if rotation is not None:
        reference = np.asarray(Image.open(path).convert("RGB").rotate(rotation, expand=True))
    np.testing.assert_array_equal(decoded, reference)


def test_decode_image_converts_embedded_icc_to_srgb(tmp_path: Path) -> None:
    arr = _gradient(16, 16)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "img.png"
    Image.fromarray(arr).save(path, icc_profile=profile)
    decoded = decode_image(str(path))
    assert decoded.shape == (16, 16, 3) and decoded.dtype == np.uint8
    # sRGB -> sRGB through littleCMS is the identity up to rounding.
    assert np.abs(decoded.astype(int) - arr.astype(int)).max() <= 1


def test_decode_image_bad_icc_falls_back_to_rgb(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    arr = _gradient(16, 16)
    path = tmp_path / "img.png"
    Image.fromarray(arr).save(path, icc_profile=b"not an icc profile")
    with caplog.at_level(logging.WARNING):
        decoded = decode_image(str(path))
    np.testing.assert_array_equal(decoded, arr)
    assert "Failed to convert image to sRGB" in caplog.text


def test_decode_image_rgba_and_gray(tmp_path: Path) -> None:
    rgba = np.dstack([_gradient(8, 8), np.full((8, 8), 128, np.uint8)])
    Image.fromarray(rgba, mode="RGBA").save(tmp_path / "rgba.png")
    assert decode_image(str(tmp_path / "rgba.png")).shape == (8, 8, 3)
    Image.fromarray(np.full((8, 8), 77, np.uint8), mode="L").save(tmp_path / "gray.png")
    gray = decode_image(str(tmp_path / "gray.png"))
    assert gray.shape == (8, 8, 3) and np.all(gray == 77)


def test_decode_image_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot decode image file"):
        decode_image(str(path))


@needs_ffmpeg
def test_encode_single_frame_is_420(tmp_path: Path) -> None:
    """Upstream (PyAV) encodes yuv420p; letting ffmpeg pick the format gave 4:4:4."""
    buf = BytesIO()
    encode_single_frame(buf, _gradient(32, 48), 33)
    path = tmp_path / "frame.mp4"
    path.write_bytes(buf.getvalue())
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=pix_fmt,profile,width,height", "-of", "compact", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "pix_fmt=yuv420p" in probe
    assert "4:4:4" not in probe
    assert "width=48" in probe and "height=32" in probe


@needs_ffmpeg
def test_preprocess_crops_odd_dims_to_even() -> None:
    """Upstream rounds the frame down to even dims before encoding and returns it cropped."""
    out = preprocess(_gradient(33, 47), crf=33)
    assert out.shape == (32, 46, 3)
    assert out.dtype == np.uint8


def test_preprocess_passthrough() -> None:
    arr = _gradient(1, 40)
    assert preprocess(arr, crf=33) is arr
    assert preprocess(_gradient(9, 9), crf=0).shape == (9, 9, 3)
