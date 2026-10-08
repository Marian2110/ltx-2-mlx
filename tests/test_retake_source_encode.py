"""Retake / extend source encode: evaluated while the encoder is loaded, tiled like upstream above one tile.

``_encode_source_video`` used to return a lazy VAE latent (``mx.synchronize`` does not evaluate a
graph), so the encode ran inside the first denoising step with the encoder weights, the source
pixels and the DiT resident together. These tests pin the evaluation order, the upstream tiling
gate (``tiled_encode(TileSizeConfig.default())`` when the source spans more than one tile), and
the per-tile evaluation in ``VideoEncoder.tiled_encode``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

import ltx_core_mlx.model.video_vae.video_vae as vv
from ltx_core_mlx.model.video_vae.tiling import TilingConfig, prepare_tiles_for_encoding
from ltx_pipelines_mlx import retake as retake_mod
from ltx_pipelines_mlx.retake import RetakePipeline


def _pipe(tmp_path) -> RetakePipeline:
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 2}}))
    return RetakePipeline(str(tmp_path), low_memory=True)


class _FakeEncoder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def encode(self, pixels):
        self.events.append("encode")
        return mx.zeros((1, 128, 1, 1, 1), dtype=mx.bfloat16) + 1

    def tiled_encode(self, pixels, tiling):
        self.events.append("tiled_encode")
        assert tiling == TilingConfig.default()
        return mx.zeros((1, 128, 1, 1, 1), dtype=mx.float32) + 1


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """A pipeline whose probe, frame loader and both encoders are stubs that log the order of events."""
    pipe = _pipe(tmp_path)
    events: list[str] = []
    info = SimpleNamespace(num_frames=49, height=768, width=512, fps=24.0, has_audio=True)
    monkeypatch.setattr(retake_mod, "probe_video_info", lambda path: info)
    # Lazy zeros of the real shape: the tiling plan reads the shape, nothing evaluates them.
    monkeypatch.setattr(retake_mod, "load_video_frames", lambda path, h, w, f: mx.zeros((1, 3, f, h, w)))
    monkeypatch.setattr(
        retake_mod, "load_audio", lambda *a, **k: SimpleNamespace(waveform=mx.zeros((1, 16000)), sample_rate=16000)
    )

    def fake_encode_audio(waveform, sample_rate, enc, proc):
        events.append("encode_audio")
        return mx.zeros((1, 8, 51, 16), dtype=mx.bfloat16) + 1

    monkeypatch.setattr(retake_mod, "encode_audio", fake_encode_audio)
    monkeypatch.setattr(pipe.image_conditioner, "load", lambda: _FakeEncoder(events))
    monkeypatch.setattr(pipe.image_conditioner, "free", lambda: events.append("free video encoder"))
    monkeypatch.setattr(pipe.audio_conditioner, "load", lambda: (object(), object()))
    monkeypatch.setattr(pipe.audio_conditioner, "free", lambda: events.append("free audio encoder"))
    real_materialize = retake_mod._materialize

    def spy_materialize(*arrays):
        events.append("materialize")
        real_materialize(*arrays)

    monkeypatch.setattr(retake_mod, "_materialize", spy_materialize)
    return pipe, info, events


def test_source_latents_are_evaluated_before_their_encoder_is_freed(stubbed):
    pipe, _, events = stubbed

    video_latent, audio_latent, meta = pipe._encode_source_video("src.mp4")

    assert events == [
        "encode",
        "materialize",
        "free video encoder",
        "encode_audio",
        "materialize",
        "free audio encoder",
    ]
    assert (video_latent.dtype, audio_latent.shape, meta.num_frames) == (mx.bfloat16, (1, 8, 51, 16), 49)


@pytest.mark.parametrize(
    ("frames", "height", "width", "expected"),
    [
        (49, 768, 512, "encode"),  # one 768 px / 80-frame tile: untiled, the latents stay exact
        (49, 1280, 704, "tiled_encode"),
        (121, 768, 512, "tiled_encode"),
    ],
)
def test_source_encode_tiles_like_upstream_above_one_tile(stubbed, frames, height, width, expected):
    """Upstream encodes the source with tiled_encode(TileSizeConfig.default()); one tile is the untiled encode."""
    pipe, info, events = stubbed
    info.num_frames, info.height, info.width = frames, height, width

    video_latent, _, _ = pipe._encode_source_video("src.mp4")

    assert events[0] == expected
    assert video_latent.dtype == mx.bfloat16  # the tiled path's fp32 blend is cast back


class _EvalSpy:
    """Stands in for ``mx`` inside video_vae: counts ``eval`` calls, delegates everything else."""

    def __init__(self) -> None:
        self.evals = 0

    def eval(self, *arrays):
        self.evals += 1
        mx.eval(*arrays)

    def __getattr__(self, name):
        return getattr(mx, name)


def test_tiled_encode_evaluates_each_tile(monkeypatch):
    """Each tile is evaluated before the next is scheduled; the blended result is unchanged."""
    video = mx.random.normal((1, 3, 17, 1024, 32), key=mx.random.key(0))
    tiling = TilingConfig.default()
    tiles = prepare_tiles_for_encoding(video.shape, tiling)
    assert len(tiles) > 1

    def fake_encode(pixels):
        # A latent that depends on the pixels, at the latent shape of the tile.
        f, h, w = (pixels.shape[2] - 1) // 8 + 1, pixels.shape[3] // 32, pixels.shape[4] // 32
        return mx.broadcast_to(mx.mean(pixels), (1, 128, f, h, w))

    encoder = SimpleNamespace(encode=fake_encode)
    expected = vv.VideoEncoder.tiled_encode(encoder, video, tiling)
    mx.eval(expected)

    spy = _EvalSpy()
    monkeypatch.setattr(vv, "mx", spy)
    out = vv.VideoEncoder.tiled_encode(encoder, video, tiling)

    assert spy.evals == len(tiles)
    assert mx.array_equal(out, expected).item()
