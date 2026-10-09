"""A2VidDistilledPipeline (``a2v --distilled``): the input audio stays frozen through both distilled stages.

The pipeline only swaps :class:`DistilledPipeline`'s two audio-state hooks, so these tests pin
(1) the hooks' default bodies to the states the distilled path built inline before, (2) the
frozen states and their survival through both denoise loops, and (3) the wiring of
``generate_and_save`` and the CLI, with the heavy stages stubbed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from ltx_core_mlx.utils.positions import compute_audio_positions
from ltx_pipelines_mlx import a2vid_distilled as a2vd_mod
from ltx_pipelines_mlx import a2vid_two_stage as a2v_mod
from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state

AUDIO_T = 13
SPATIAL = (2, 4, 4)


def _write_pack(tmp_path, *, ltx25: bool) -> str:
    transformer: dict = {"num_layers": 48}
    if ltx25:
        transformer["ff_bias"] = False
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))
    return str(tmp_path)


def _source_tokens(seed: int = 3) -> mx.array:
    return mx.random.normal((1, AUDIO_T, 128), key=mx.random.key(seed)).astype(mx.bfloat16)


def _assert_same_state(a, b) -> None:
    for field in ("latent", "clean_latent", "denoise_mask", "positions"):
        x, y = getattr(a, field), getattr(b, field)
        assert x.dtype == y.dtype, field
        assert mx.array_equal(x, y).item(), field
    assert a.frozen == b.frozen


def _assert_frozen_on(state, tokens: mx.array) -> None:
    assert state.frozen
    assert mx.array_equal(state.latent, tokens).item()
    assert mx.array_equal(state.clean_latent, tokens).item()
    assert not mx.any(state.denoise_mask).item()


# --- the hooks ----------------------------------------------------------------------------------


def test_default_hooks_build_the_states_the_distilled_path_built_inline(tmp_path):
    """Refactor guard: the default hook bodies are the old inline ``create_noised_state`` calls."""
    pipe = DistilledPipeline(model_dir=_write_pack(tmp_path, ltx25=True))
    positions = compute_audio_positions(AUDIO_T)
    stage1_out = _source_tokens(5)

    _assert_same_state(
        pipe._stage1_audio_state((1, AUDIO_T, 128), positions, SPATIAL, 43),
        create_noised_state(
            base_shape=(1, AUDIO_T, 128),
            conditionings=[],
            spatial_dims=SPATIAL,
            positions=positions,
            seed=43,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        ),
    )
    _assert_same_state(
        pipe._stage2_audio_state(stage1_out, positions, SPATIAL, 44, 0.85),
        create_noised_state(
            base_shape=stage1_out.shape,
            conditionings=[],
            spatial_dims=SPATIAL,
            positions=positions,
            seed=44,
            sigma=0.85,
            initial_latent=stage1_out,
        ),
    )


def test_a2v_hooks_without_source_audio_fall_back_to_the_parent(tmp_path):
    a2v = A2VidDistilledPipeline(model_dir=_write_pack(tmp_path, ltx25=True))
    parent = DistilledPipeline(model_dir=str(tmp_path))
    positions = compute_audio_positions(AUDIO_T)
    stage1_out = _source_tokens(5)

    _assert_same_state(
        a2v._stage1_audio_state((1, AUDIO_T, 128), positions, SPATIAL, 43),
        parent._stage1_audio_state((1, AUDIO_T, 128), positions, SPATIAL, 43),
    )
    _assert_same_state(
        a2v._stage2_audio_state(stage1_out, positions, SPATIAL, 44, 0.85),
        parent._stage2_audio_state(stage1_out, positions, SPATIAL, 44, 0.85),
    )


def test_a2v_hooks_freeze_the_source_tokens_in_both_stages(tmp_path):
    pipe = A2VidDistilledPipeline(model_dir=_write_pack(tmp_path, ltx25=True))
    tokens = _source_tokens()
    pipe._source_audio_tokens = tokens
    positions = compute_audio_positions(AUDIO_T)

    state1 = pipe._stage1_audio_state((1, AUDIO_T, 128), positions, SPATIAL, 43)
    # Stage 2 re-attaches the source track: whatever stage 1 returned and whatever sigma is ignored.
    state2 = pipe._stage2_audio_state(_source_tokens(9), positions, SPATIAL, 44, 0.85)

    _assert_frozen_on(state1, tokens)
    _assert_frozen_on(state2, tokens)
    assert mx.array_equal(state1.positions, positions).item()


def test_a2v_hooks_reject_a_track_that_does_not_cover_the_clip(tmp_path):
    pipe = A2VidDistilledPipeline(model_dir=_write_pack(tmp_path, ltx25=True))
    pipe._source_audio_tokens = _source_tokens()[:, :-1]

    with pytest.raises(ValueError, match="Encoded audio has shape"):
        pipe._stage1_audio_state((1, AUDIO_T, 128), compute_audio_positions(AUDIO_T), SPATIAL, 43)


@pytest.mark.parametrize("ancestral", [False, True], ids=["euler", "ancestral"])
def test_frozen_audio_comes_out_of_both_loops_bit_identical(tmp_path, ancestral):
    """The distilled stage loops keep a frozen audio stream untouched and condition it on sigma 0."""
    pipe = A2VidDistilledPipeline(model_dir=_write_pack(tmp_path, ltx25=ancestral))
    tokens = _source_tokens()
    pipe._source_audio_tokens = tokens
    audio_state = pipe._stage1_audio_state((1, AUDIO_T, 128), compute_audio_positions(AUDIO_T), SPATIAL, 43)
    video_state = create_noised_state(
        base_shape=(1, 16, 128),
        conditionings=[],
        spatial_dims=(1, 4, 4),
        positions=mx.zeros((1, 16, 3)),
        seed=42,
        sigma=1.0,
    )
    calls: list[dict] = []

    def model(**kwargs):
        calls.append(kwargs)
        # A prediction far from the input track: only the frozen mask can keep the tokens.
        return kwargs["video_latent"] * 0.5, mx.full(kwargs["audio_latent"].shape, 7.0, dtype=mx.bfloat16)

    out = pipe._run_denoise_loop(
        model=model,
        video_state=video_state,
        audio_state=audio_state,
        video_text_embeds=mx.zeros((1, 8, 4096)),
        audio_text_embeds=mx.zeros((1, 8, 2048)),
        sigmas=[1.0, 0.7, 0.4, 0.0],
        video_cross_attention_mask=None,
        on_step=None,
        seed=42,
        ancestral=ancestral,
        noise_seed_offset=10000,
    )

    assert mx.array_equal(out.audio_latent, tokens).item()
    assert len(calls) == 3
    for kwargs in calls:
        assert not mx.any(kwargs["audio_sigma"]).item()
        assert not mx.any(kwargs["audio_timesteps"]).item()


# --- generate_and_save ---------------------------------------------------------------------------


class _StopError(Exception):
    """Raised by a stub to end a run at a known point."""


@pytest.fixture
def stubbed_pipe(tmp_path, monkeypatch):
    """A 2.5-pack pipeline whose model stages are stubs that record the audio states they get."""
    pipe = A2VidDistilledPipeline(model_dir=_write_pack(tmp_path, ltx25=True))
    tokens = _source_tokens()
    record: dict = {"states": []}

    def fake_encode(p, audio_path, **kwargs):
        record["encode"] = dict(audio_path=audio_path, **kwargs)
        return tokens

    def fake_generate_two_stage(**kwargs):
        record["generate"] = kwargs
        positions = compute_audio_positions(AUDIO_T)
        record["states"].append(pipe._stage1_audio_state((1, AUDIO_T, 128), positions, SPATIAL, kwargs["seed"] + 1))
        record["states"].append(
            pipe._stage2_audio_state(_source_tokens(9), positions, SPATIAL, kwargs["seed"] + 2, 0.9)
        )
        return mx.zeros((1, 128, 2, 4, 4)), mx.zeros((1, 8, AUDIO_T, 16))

    def fake_decode(p, video_latent, output_path, **kwargs):
        record["decode"] = dict(output_path=output_path, **kwargs)

    monkeypatch.setattr(a2vd_mod, "encode_source_audio", fake_encode)
    monkeypatch.setattr(a2vd_mod, "decode_with_source_audio", fake_decode)
    monkeypatch.setattr(pipe, "generate_two_stage", fake_generate_two_stage)
    monkeypatch.setattr(pipe, "_load_decoders", lambda: None)
    return pipe, tokens, record


def test_generate_and_save_freezes_the_input_track_and_muxes_it(stubbed_pipe, tmp_path):
    pipe, tokens, record = stubbed_pipe

    out = pipe.generate_and_save(
        prompt="a woman sings",
        output_path=str(tmp_path / "out.mp4"),
        audio_path="vocals.wav",
        height=256,
        width=384,
        num_frames=50,
        frame_rate=24.0,
        seed=7,
        stage1_steps=4,
        images=["anchor"],
        audio_start_time=1.5,
    )

    assert out == str(tmp_path / "out.mp4")
    # 50 frames are floored to the 8k+1 grid before the audio is encoded and the clip is sized.
    assert record["encode"] == dict(
        audio_path="vocals.wav", num_frames=49, frame_rate=24.0, start_time=1.5, max_duration=49 / 24.0
    )
    gen = record["generate"]
    assert (gen["num_frames"], gen["seed"], gen["stage1_steps"], gen["images"]) == (49, 7, 4, ["anchor"])
    for state in record["states"]:
        _assert_frozen_on(state, tokens)
    assert record["decode"] == dict(
        output_path=str(tmp_path / "out.mp4"), audio_path="vocals.wav", start_time=1.5, num_frames=49, frame_rate=24.0
    )
    assert pipe._source_audio_tokens is None


def test_generate_and_save_clears_the_track_when_generation_fails(stubbed_pipe, monkeypatch, tmp_path):
    pipe, _, _ = stubbed_pipe

    def boom(**kwargs):
        raise _StopError

    monkeypatch.setattr(pipe, "generate_two_stage", boom)
    with pytest.raises(_StopError):
        pipe.generate_and_save(
            prompt="x", output_path=str(tmp_path / "o.mp4"), audio_path="a.wav", num_frames=9, frame_rate=24.0
        )
    assert pipe._source_audio_tokens is None


def test_audio_path_is_required(tmp_path):
    pipe = A2VidDistilledPipeline(model_dir=_write_pack(tmp_path, ltx25=True))
    with pytest.raises(ValueError, match="audio_path is required"):
        pipe.generate_and_save(prompt="x", output_path="o.mp4", num_frames=9, frame_rate=24.0)


def test_short_audio_fails_before_any_model_work(tmp_path, monkeypatch):
    pipe = A2VidDistilledPipeline(model_dir=_write_pack(tmp_path, ltx25=True))
    monkeypatch.setattr(pipe, "_load_audio_encoder", lambda: None)
    pipe.audio_encoder = object()
    pipe.audio_processor = object()
    monkeypatch.setattr(
        a2v_mod, "load_audio", lambda *a, **k: SimpleNamespace(waveform=mx.zeros((1, 16000)), sample_rate=16000)
    )
    # 9 frames at 24 fps need 9 audio latent frames; give it 8.
    monkeypatch.setattr(a2v_mod, "encode_audio", lambda *a, **k: mx.zeros((1, 8, 8, 16), dtype=mx.bfloat16))

    def must_not_run(**kwargs):
        raise AssertionError("the length check must fire before the prompt is encoded")

    monkeypatch.setattr(pipe, "generate_two_stage", must_not_run)
    with pytest.raises(ValueError, match="Audio is too short"):
        pipe.generate_and_save(
            prompt="x", output_path="o.mp4", audio_path="a.wav", height=128, width=128, num_frames=9, frame_rate=24.0
        )


def test_decode_with_source_audio_trims_the_track_and_removes_the_temp_file(monkeypatch, tmp_path):
    """The muxed track is the input waveform cut to the clip; the temp wav goes even if decoding fails."""
    saved: dict = {}
    monkeypatch.setattr(
        a2v_mod,
        "load_audio",
        lambda *a, **k: SimpleNamespace(waveform=mx.ones((1, 2, 48000 * 3)), sample_rate=48000),
    )

    class _Decoder:
        def decode_and_stream(self, video_latent, output_path, *, frame_rate, audio_path):
            saved["audio_path"] = audio_path
            raise _StopError

    def save_waveform(waveform, path, sample_rate):
        saved["samples"] = waveform.shape[-1]
        open(path, "wb").close()

    pipe = SimpleNamespace(video_decoder_block=_Decoder(), _save_waveform=save_waveform)
    with pytest.raises(_StopError):
        a2v_mod.decode_with_source_audio(
            pipe,
            mx.zeros((1, 128, 2, 4, 4)),
            str(tmp_path / "o.mp4"),
            audio_path="a.wav",
            start_time=0.0,
            num_frames=49,
            frame_rate=24.0,
        )

    assert saved["samples"] == int(49 / 24.0 * 48000)
    assert not (tmp_path / saved["audio_path"]).exists()


# --- CLI -----------------------------------------------------------------------------------------


def _a2v_args(tmp_path, *extra: str):
    from ltx_pipelines_mlx import cli

    return cli._build_parser().parse_args(
        ["a2v", "-p", "a singer", "--audio", "song.wav", "--frame-rate", "24", "-o", str(tmp_path / "out.mp4"), "-q"]
        + list(extra)
    )


def test_cli_a2v_distilled_builds_the_distilled_pipeline(monkeypatch, tmp_path):
    from ltx_pipelines_mlx import cli

    received: dict = {}

    class _FakePipe:
        def __init__(self, **kwargs) -> None:
            received["init"] = kwargs

        def generate_and_save(self, **kwargs) -> str:
            received.update(kwargs)
            return kwargs["output_path"]

    monkeypatch.setattr(a2vd_mod, "A2VidDistilledPipeline", _FakePipe)
    monkeypatch.setattr(cli, "_print_result", lambda *a, **k: None)
    cli._cmd_a2v(_a2v_args(tmp_path, "--distilled", "--stage1-steps", "4", "--low-ram", "-i", "photo.png"))

    assert received["init"]["low_ram_streaming"] is True
    assert received["stage1_steps"] == 4
    assert received["audio_path"] == "song.wav"
    assert not {"cfg_scale", "stg_scale", "negative_prompt", "enable_teacache"} & received.keys()


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (("--cfg-scale", "3"), "--cfg-scale"),
        (("--stg-scale", "1"), "--stg-scale"),
        (("--negative-prompt", "blurry"), "--negative-prompt"),
        (("--enable-teacache",), "TeaCache"),
    ],
)
def test_cli_a2v_distilled_rejects_cfg_flags_before_building_anything(monkeypatch, tmp_path, flags, message):
    from ltx_pipelines_mlx import cli

    def must_not_build(**kwargs):
        raise AssertionError("the flags must be rejected before the pipeline is built")

    monkeypatch.setattr(a2vd_mod, "A2VidDistilledPipeline", must_not_build)
    with pytest.raises(SystemExit, match=message):
        cli._cmd_a2v(_a2v_args(tmp_path, "--distilled", *flags))


def test_encode_source_audio_evaluates_the_tokens_before_freeing_the_encoder(monkeypatch):
    """The tokens are materialised while the audio encoder is still loaded, then the encoder is freed."""
    events: list[str] = []
    monkeypatch.setattr(
        a2v_mod, "load_audio", lambda *a, **k: SimpleNamespace(waveform=mx.zeros((1, 16000)), sample_rate=16000)
    )
    monkeypatch.setattr(a2v_mod, "encode_audio", lambda *a, **k: mx.zeros((1, 8, 16, 16), dtype=mx.bfloat16))
    real_eval = a2v_mod.mx.eval

    def spy_eval(*arrays):
        events.append("eval")
        return real_eval(*arrays)

    monkeypatch.setattr(a2v_mod.mx, "eval", spy_eval)
    pipe = SimpleNamespace(
        _load_audio_encoder=lambda: None,
        audio_encoder=object(),
        audio_processor=object(),
        audio_patchifier=SimpleNamespace(patchify=lambda latent: (latent.reshape(1, -1, 128), None)),
        low_memory=True,
        audio_conditioner=SimpleNamespace(free=lambda: events.append("free")),
    )

    tokens = a2v_mod.encode_source_audio(pipe, "a.wav", num_frames=9, frame_rate=24.0, start_time=0.0, max_duration=1.0)

    assert events == ["eval", "free"]
    assert tokens.shape[0] == 1
