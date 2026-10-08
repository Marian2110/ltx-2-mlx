"""RetakePipeline(distilled=True) (``retake --distilled``): upstream's default retake mode.

The distilled branch encodes the prompt alone, loads the distilled transformer and runs the
plain Euler loop on the distilled sigma table. These tests pin (1) that it never touches the
dev transformer, the negative prompt or the guided loop, (2) that the temporal mask keeps
every token outside ``[start, end)`` (and a frozen audio stream) bit-identical through the
real loop, (3) the step schedule, (4) that the default path still runs dev + CFG, and (5)
the CLI wiring, with the CFG flags rejected before anything is built.
"""

from __future__ import annotations

import json
from typing import ClassVar

import mlx.core as mx
import pytest

from ltx_pipelines_mlx import retake as retake_mod
from ltx_pipelines_mlx.retake import RetakePipeline
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

# 17 pixel frames at 64 x 64 -> 3 latent frames of 2 x 2 tokens.
NUM_FRAMES, HEIGHT, WIDTH = 17, 64, 64
F, H, W = 3, 2, 2
AUDIO_T = 12


def _write_pack(tmp_path, *files: str) -> str:
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 2}}))
    for name in files:
        (tmp_path / name).write_bytes(b"")
    return str(tmp_path)


def _source(seed: int = 3) -> tuple[mx.array, mx.array]:
    video = mx.random.normal((1, 128, F, H, W), key=mx.random.key(seed)).astype(mx.bfloat16)
    audio = mx.random.normal((1, 8, AUDIO_T, 16), key=mx.random.key(seed + 1)).astype(mx.bfloat16)
    return video, audio


class _LoopSpy:
    """Stand-in for a denoising loop: records kwargs, echoes the input latents."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return DenoiseOutput(video_latent=kwargs["video_state"].latent, audio_latent=kwargs["audio_state"].latent)


def _must_not_run(name: str):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{name} must not run on the distilled retake path")

    return _fail


class _FakeModel:
    """X0 model stand-in: predicts a constant far from the source, records the calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (
            mx.full(kwargs["video_latent"].shape, 7.0, dtype=mx.bfloat16),
            mx.full(kwargs["audio_latent"].shape, -7.0, dtype=mx.bfloat16),
        )


@pytest.fixture
def distilled_pipe(tmp_path, monkeypatch):
    """A distilled RetakePipeline whose text encoder and transformer loader are stubs."""
    pipe = RetakePipeline(_write_pack(tmp_path, "transformer-dev.safetensors"), low_memory=False, distilled=True)
    record: dict = {"prompts": [], "loaded": []}

    def fake_encode(prompt):
        record["prompts"].append(prompt)
        return mx.zeros((1, 4, 4096), dtype=mx.bfloat16), mx.zeros((1, 4, 2048), dtype=mx.bfloat16)

    model = _FakeModel()

    def fake_load(path):
        record["loaded"].append(path)
        return model

    monkeypatch.setattr(pipe, "_load_text_encoder", lambda: None)
    monkeypatch.setattr(pipe, "_encode_text", fake_encode)
    monkeypatch.setattr(pipe, "_load_transformer_with_optional_streaming", fake_load)
    monkeypatch.setattr(pipe, "_encode_text_with_negative", _must_not_run("_encode_text_with_negative"))
    monkeypatch.setattr(pipe, "_load_dev_transformer", _must_not_run("_load_dev_transformer"))
    monkeypatch.setattr(retake_mod, "guided_denoise_loop", _must_not_run("guided_denoise_loop"))
    monkeypatch.setattr(retake_mod, "X0Model", lambda dit: dit)
    return pipe, model, record


def _retake(pipe, **overrides):
    video, audio = _source()
    kwargs = dict(
        prompt="she waves",
        source_video_latent=video,
        source_audio_latent=audio,
        start_frame=1,
        end_frame=2,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        frame_rate=24.0,
        seed=7,
    )
    kwargs.update(overrides)
    return pipe.retake(**kwargs)


# --- the distilled branch -------------------------------------------------------------------------


def test_distilled_retake_skips_the_dev_transformer_negative_prompt_and_guided_loop(distilled_pipe, tmp_path):
    pipe, model, record = distilled_pipe
    (tmp_path / "transformer-distilled-1.1.safetensors").write_bytes(b"")

    _retake(pipe)

    assert record["prompts"] == ["she waves"]  # the prompt alone, no negative
    assert record["loaded"] == [tmp_path / "transformer-distilled-1.1.safetensors"]
    assert len(model.calls) == len(DISTILLED_SIGMAS) - 1  # one forward per step
    assert pipe.text_encoder is None


def test_distilled_retake_prefers_transformer_safetensors(distilled_pipe, tmp_path):
    """Same resolution order as DistilledPipeline.load: transformer.safetensors, then transformer-distilled*."""
    pipe, _, record = distilled_pipe
    (tmp_path / "transformer.safetensors").write_bytes(b"")
    (tmp_path / "transformer-distilled.safetensors").write_bytes(b"")

    _retake(pipe)

    assert record["loaded"] == [tmp_path / "transformer.safetensors"]


def test_distilled_retake_rejects_a_negative_prompt_before_encoding(distilled_pipe):
    pipe, _, record = distilled_pipe
    with pytest.raises(ValueError, match="negative_prompt requires a CFG pipeline"):
        _retake(pipe, negative_prompt="blurry")
    assert record["prompts"] == [] and record["loaded"] == []


@pytest.mark.parametrize(
    ("num_steps", "sigmas"),
    [
        (30, DISTILLED_SIGMAS),  # the API default: the whole table
        (8, DISTILLED_SIGMAS),
        (3, [1.0, 0.725, 0.421875, 0.0]),
    ],
)
def test_distilled_retake_runs_the_distilled_table(distilled_pipe, monkeypatch, num_steps, sigmas):
    pipe, _, _ = distilled_pipe
    loop = _LoopSpy()
    monkeypatch.setattr(retake_mod, "denoise_loop", loop)

    _retake(pipe, num_steps=num_steps)

    assert len(loop.calls) == 1
    assert loop.calls[0]["sigmas"] == sigmas
    assert "video_guider_factory" not in loop.calls[0]


def test_distilled_retake_keeps_tokens_outside_the_window_bit_identical(distilled_pipe):
    """Through the real Euler loop: the source survives outside [start, end), the window is regenerated."""
    pipe, model, _ = distilled_pipe
    video, audio = _source()

    out_video, out_audio = _retake(pipe)

    # The loop returns float32 (per-token sigmas promote it); the kept values are the bf16 source's exactly.
    for f in (0, 2):
        assert mx.array_equal(out_video[:, :, f], video[:, :, f]).item(), f"latent frame {f}"
    assert not mx.array_equal(out_video[:, :, 1], video[:, :, 1]).item()

    # Audio follows the same window (audio_T / F tokens per latent frame), the rest is kept.
    start, end = round(1 * AUDIO_T / F), round(2 * AUDIO_T / F)
    assert mx.array_equal(out_audio[:, :, :start], audio[:, :, :start]).item()
    assert mx.array_equal(out_audio[:, :, end:], audio[:, :, end:]).item()
    assert not mx.array_equal(out_audio[:, :, start:end], audio[:, :, start:end]).item()

    # The window is denoised at the per-token timesteps sigma * mask, the rest at 0.
    timesteps = model.calls[0]["video_timesteps"]
    assert mx.array_equal(timesteps[:, : H * W], mx.zeros_like(timesteps[:, : H * W])).item()
    assert mx.all(timesteps[:, H * W : 2 * H * W] == 1.0).item()


def test_distilled_retake_no_regen_audio_freezes_the_audio(distilled_pipe, monkeypatch):
    pipe, model, _ = distilled_pipe
    loop = _LoopSpy()
    monkeypatch.setattr(retake_mod, "denoise_loop", loop)
    _retake(pipe, regenerate_audio=False)
    audio_state = loop.calls[0]["audio_state"]
    assert audio_state.frozen
    assert not mx.any(audio_state.denoise_mask).item()
    assert loop.calls[0]["video_state"].frozen is False


def test_distilled_retake_no_regen_audio_returns_the_source_audio(distilled_pipe):
    """A frozen audio stream comes out of the real loop bit-identical and is conditioned on sigma 0."""
    pipe, model, _ = distilled_pipe
    _, audio = _source()

    _, out_audio = _retake(pipe, regenerate_audio=False)

    assert mx.array_equal(out_audio, audio).item()
    for kwargs in model.calls:
        assert not mx.any(kwargs["audio_sigma"]).item()


def test_extend_refuses_the_distilled_mode(distilled_pipe):
    pipe, _, record = distilled_pipe
    video, audio = _source()
    with pytest.raises(NotImplementedError, match="extend has no distilled mode"):
        pipe.extend(
            prompt="continue",
            source_video_latent=video,
            source_audio_latent=audio,
            extend_frames=1,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=24.0,
        )
    assert record["prompts"] == []


# --- the default path ----------------------------------------------------------------------------


def test_default_retake_still_runs_dev_cfg(tmp_path, monkeypatch):
    """Guards the default path: dev transformer, negative prompt, guided loop on the dynamic schedule."""
    pipe = RetakePipeline(_write_pack(tmp_path, "transformer-dev.safetensors"), low_memory=False)
    assert pipe.distilled is False
    negatives = (mx.ones((1, 4, 4096), dtype=mx.bfloat16), mx.ones((1, 4, 2048), dtype=mx.bfloat16))
    record: dict = {}

    def fake_encode_with_negative(prompt, negative_prompt=None):
        record["encode"] = (prompt, negative_prompt)
        return mx.zeros((1, 4, 4096), dtype=mx.bfloat16), mx.zeros((1, 4, 2048), dtype=mx.bfloat16), *negatives

    def fake_load(path):
        record["loaded"] = path
        return object()

    monkeypatch.setattr(pipe, "_encode_text_with_negative", fake_encode_with_negative)
    monkeypatch.setattr(pipe, "_load_transformer_with_optional_streaming", fake_load)
    monkeypatch.setattr(pipe, "_encode_text", _must_not_run("_encode_text (positive-only)"))
    monkeypatch.setattr(retake_mod, "X0Model", lambda dit: dit)
    monkeypatch.setattr(retake_mod, "denoise_loop", _must_not_run("denoise_loop"))
    guided = _LoopSpy()
    monkeypatch.setattr(retake_mod, "guided_denoise_loop", guided)

    _retake(pipe, num_steps=4, negative_prompt="blurry", cfg_scale=2.5, stg_scale=0.5)

    assert record["encode"] == ("she waves", "blurry")
    assert record["loaded"] == tmp_path / "transformer-dev.safetensors"
    assert len(guided.calls) == 1
    call = guided.calls[0]
    assert call["sigmas"] == ltx2_schedule(4, num_tokens=F * H * W)
    video_factory, audio_factory = call["video_guider_factory"], call["audio_guider_factory"]
    assert video_factory.negative_context is negatives[0]
    assert audio_factory.negative_context is negatives[1]
    video_params, audio_params = video_factory.params(1.0), audio_factory.params(1.0)
    assert (video_params.cfg_scale, video_params.stg_scale) == (2.5, 0.5)
    assert (audio_params.cfg_scale, audio_params.stg_scale) == (7.0, 0.5)
    for params in (video_params, audio_params):
        assert (params.rescale_scale, params.modality_scale, params.stg_blocks) == (0.7, 3.0, [28])


# --- CLI -----------------------------------------------------------------------------------------


def _retake_args(tmp_path, *extra: str):
    from ltx_pipelines_mlx import cli

    return cli._build_parser().parse_args(
        ["retake", "-p", "she waves", "-v", "src.mp4", "--start", "2", "--end", "5", "-o", str(tmp_path / "o.mp4")]
        + ["-q", *extra]
    )


class _FakeCliPipe:
    received: ClassVar[dict] = {}

    def __init__(self, **kwargs) -> None:
        type(self).received = {"init": kwargs}

    def retake_from_video(self, **kwargs):
        type(self).received.update(kwargs)
        return None, None


@pytest.mark.parametrize(
    ("extra", "distilled"),
    [(("--distilled", "--steps", "4", "--low-ram", "--no-regen-audio"), True), ((), False)],
    ids=["distilled", "default"],
)
def test_cli_retake_builds_the_requested_mode(monkeypatch, tmp_path, extra, distilled):
    from ltx_pipelines_mlx import cli

    monkeypatch.setattr(retake_mod, "RetakePipeline", _FakeCliPipe)
    monkeypatch.setattr(cli, "_decode_and_save", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_print_result", lambda *a, **k: None)

    cli._cmd_retake(_retake_args(tmp_path, *extra))

    received = _FakeCliPipe.received
    assert received["init"]["distilled"] is distilled
    assert (received["start_frame"], received["end_frame"]) == (2, 5)
    if distilled:
        assert received["init"]["low_ram_streaming"] is True
        assert received["num_steps"] == 4
        assert received["regenerate_audio"] is False
    assert not {"cfg_scale", "stg_scale", "negative_prompt"} & received.keys()


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (("--cfg-scale", "3"), "--cfg-scale"),
        (("--stg-scale", "1"), "--stg-scale"),
        (("--negative-prompt", "blurry"), "--negative-prompt"),
        (("--negative-prompt", ""), "--negative-prompt"),
    ],
)
def test_cli_retake_distilled_rejects_cfg_flags_before_building_anything(monkeypatch, tmp_path, flags, message):
    from ltx_pipelines_mlx import cli

    monkeypatch.setattr(retake_mod, "RetakePipeline", _must_not_run("RetakePipeline()"))
    with pytest.raises(SystemExit, match=message):
        cli._cmd_retake(_retake_args(tmp_path, "--distilled", *flags))
