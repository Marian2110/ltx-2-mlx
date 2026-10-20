"""Prompt Relay on the audio stream (``audio_attn2``).

The reference ComfyUI node (kijai/ComfyUI-PromptRelay) masks both cross-attentions of every LTX block:
``attn2`` with video queries on whole latent frames, ``audio_attn2`` with audio queries placed at
``i * latent_frames / Na`` (``build_temporal_cost_scaled``). These tests pin the audio mask against a
transcription of that function, its path from the pipelines through the four sampler loops and the model
to ``audio_attn2``, and the cases where it must not apply: no ``--segment``, ``audio_strength=0``, the
negative (CFG) pass and a frozen audio stream. Pure CPU, tiny modules, no weights.
"""

from __future__ import annotations

import json
import math

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.conditioning.prompt_relay import (
    PromptRelayInput,
    build_relay_mask,
    build_relay_mask_audio,
    distribute_segment_lengths,
)
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

# --- the mask ------------------------------------------------------------------------------------


def _reference_scaled_cost(token_ranges, segment_lengths, num_audio, num_text, latent_frames, epsilon, strength):
    """kijai/ComfyUI-PromptRelay ``build_segments`` + ``build_temporal_cost_scaled``, in numpy."""
    sigma = 1.0 / math.log(1.0 / epsilon)
    offset = np.zeros((num_audio, num_text), dtype=np.float32)
    query_frames = np.arange(num_audio, dtype=np.float32) * latent_frames / num_audio
    cursor = 0
    for (start, end), length in zip(token_ranges, segment_lengths):
        midpoint = (2 * cursor + length) // 2
        window = max(length // 2 - 2, 0)
        d = np.abs(query_frames[:, None] - midpoint)
        offset[:, start:end] = strength * np.maximum(d - window, 0.0) ** 2 / (2 * sigma**2)
        cursor += length
    return -offset


@pytest.mark.parametrize(
    ("latent_frames", "num_audio", "segments", "epsilon", "strength"),
    [
        (13, 101, [7, 6], 1e-3, 1.0),  # 97 frames @ 24 fps: the e2e shape
        (7, 51, [3, 4], 1e-3, 1.0),  # 49 frames
        (13, 101, [4, 4, 5], 0.05, 0.5),
        (31, 251, [10, 11, 10], 1e-3, 2.0),  # 241 frames
    ],
)
def test_audio_mask_matches_the_reference_scaled_cost(latent_frames, num_audio, segments, epsilon, strength):
    ranges = [(10 + 5 * i, 15 + 5 * i) for i in range(len(segments))]
    mask = build_relay_mask_audio(
        token_ranges=ranges,
        segment_lengths=segments,
        num_audio_tokens=num_audio,
        latent_frames=latent_frames,
        num_text_tokens=64,
        epsilon=epsilon,
        strength=strength,
        dtype=mx.float32,
    )
    assert mask.shape == (1, 1, num_audio, 64)
    expected = _reference_scaled_cost(ranges, segments, num_audio, 64, latent_frames, epsilon, strength)
    np.testing.assert_allclose(np.array(mask)[0, 0], expected, rtol=1e-6, atol=1e-6)


def test_audio_mask_dtype_and_free_columns():
    mask = build_relay_mask_audio(
        token_ranges=[(3, 5), (5, 8)],
        segment_lengths=[7, 6],
        num_audio_tokens=101,
        latent_frames=13,
        num_text_tokens=16,
    )
    assert mask.dtype == mx.bfloat16
    m = np.array(mask.astype(mx.float32))[0, 0]
    # Global prompt (0-2) and register columns (8+) are never penalised.
    assert not m[:, :3].any() and not m[:, 8:].any()
    # Audio token 0 sits in segment 0: its own words are close to free, the other segment's are far off.
    assert m[0, 3] > m[0, 5]
    # The last token sits in segment 1: the reverse.
    assert m[-1, 5] > m[-1, 3]
    # At segment 0's midpoint (latent frame 3 -> audio token 3 * 101 / 13 ~ 23) its columns are free.
    assert m[23, 3] == 0.0 and m[23, 5] < -100.0


def test_audio_mask_equals_video_mask_with_one_token_per_frame():
    """Same segments as the video mask: with Na == F the scaled positions are the integer frames."""
    kw = dict(token_ranges=[(2, 4), (4, 6)], segment_lengths=[5, 8], latent_frames=13, num_text_tokens=8)
    audio = build_relay_mask_audio(num_audio_tokens=13, **kw)
    video = build_relay_mask(num_video_tokens=13, tokens_per_frame=1, **kw)
    assert mx.array_equal(audio, video).item()


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        (dict(segment_lengths=[13, 0]), "zero-length"),
        (dict(token_ranges=[(2, 4), (4, 9)]), "exceed the text axis"),
        (dict(epsilon=1.0), "epsilon"),
        (dict(num_audio_tokens=0), "at least one audio token"),
    ],
)
def test_audio_mask_rejects_what_the_video_mask_rejects(kw, match):
    args = dict(token_ranges=[(2, 4), (4, 6)], segment_lengths=[7, 6], num_audio_tokens=101, latent_frames=13)
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        build_relay_mask_audio(num_text_tokens=8, **args)


def test_relay_input_validates_audio_knobs_up_front():
    assert PromptRelayInput(["a"]).audio_strength is None  # the pipeline picks
    assert PromptRelayInput(["a"]).audio_epsilon is None
    with pytest.raises(ValueError, match="audio strength"):
        PromptRelayInput(["a"], audio_strength=-1.0)
    with pytest.raises(ValueError, match="epsilon"):
        PromptRelayInput(["a"], audio_epsilon=1.5)


# --- the pipeline builder ------------------------------------------------------------------------


def _audio(num_tokens: int = 101, frozen: bool = False) -> LatentState:
    mask = mx.zeros if frozen else mx.ones
    return LatentState(
        mx.zeros((1, num_tokens, 8)), mx.zeros((1, num_tokens, 8)), mask((1, num_tokens, 1)), frozen=frozen
    )


def _audio_builder(relay, ranges=((2, 4), (4, 6)), default=1.0):
    from ltx_pipelines_mlx._base import BasePipeline

    return BasePipeline._prompt_relay_audio_mask_builder(
        relay, None if ranges is None else list(ranges), 8, default_audio_strength=default
    )


def test_builder_is_off_without_relay_or_with_zero_audio_strength():
    assert _audio_builder(None, ranges=None)(13, _audio()) is None
    assert _audio_builder(PromptRelayInput(["a", "b"], audio_strength=0.0))(13, _audio()) is None


def test_builder_builds_nothing_for_a_frozen_or_absent_audio_stream(monkeypatch):
    from ltx_core_mlx.conditioning import prompt_relay

    def must_not_build(*a, **k):
        raise AssertionError("no audio mask may be built for a frozen / absent stream")

    monkeypatch.setattr(prompt_relay, "build_relay_mask_audio", must_not_build)
    build = _audio_builder(PromptRelayInput(["a", "b"]))
    assert build(13, _audio(frozen=True)) is None
    assert build(13, None) is None


def test_builder_uses_the_video_segments_and_its_own_knobs():
    relay = PromptRelayInput(["a", "b"], segment_lengths=[5, None], epsilon=1e-3, audio_strength=0.5)
    got = _audio_builder(relay)(13, _audio())
    lengths = distribute_segment_lengths(2, 13, [5, None])
    expected = build_relay_mask_audio([(2, 4), (4, 6)], lengths, 101, 13, 8, epsilon=1e-3, strength=0.5)
    assert mx.array_equal(got, expected).item()


def test_builder_audio_epsilon_overrides_and_otherwise_inherits():
    inherit = _audio_builder(PromptRelayInput(["a", "b"], epsilon=0.1))(13, _audio())
    own = _audio_builder(PromptRelayInput(["a", "b"], epsilon=0.1, audio_epsilon=1e-3))(13, _audio())
    lengths = [7, 6]
    assert mx.array_equal(inherit, build_relay_mask_audio([(2, 4), (4, 6)], lengths, 101, 13, 8, epsilon=0.1)).item()
    assert mx.array_equal(own, build_relay_mask_audio([(2, 4), (4, 6)], lengths, 101, 13, 8, epsilon=1e-3)).item()


# --- block and model -----------------------------------------------------------------------------


def _block() -> BasicAVTransformerBlock:
    return BasicAVTransformerBlock(
        video_dim=32,
        audio_dim=16,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=4,
        av_cross_head_dim=4,
    )


def _block_kwargs(nv: int, na: int, nt: int) -> dict:
    # Nonzero text cross-attention gates (AdaLN chunk 8), else attn2 / audio_attn2 contribute nothing.
    video_adaln = mx.zeros((1, 9 * 32)).at[:, 8 * 32 :].add(1.0)
    audio_adaln = mx.zeros((1, 9 * 16)).at[:, 8 * 16 :].add(1.0)
    return dict(
        video_hidden=mx.random.normal((1, nv, 32)),
        audio_hidden=mx.random.normal((1, na, 16)),
        video_adaln_params=video_adaln,
        audio_adaln_params=audio_adaln,
        video_prompt_adaln_params=mx.zeros((1, 2 * 32)),
        audio_prompt_adaln_params=mx.zeros((1, 2 * 16)),
        av_ca_video_params=mx.zeros((1, 4 * 32)),
        av_ca_audio_params=mx.zeros((1, 4 * 16)),
        av_ca_a2v_gate_params=mx.zeros((1, 32)),
        av_ca_v2a_gate_params=mx.zeros((1, 16)),
        video_text_embeds=mx.random.normal((1, nt, 32)),
        audio_text_embeds=mx.random.normal((1, nt, 16)),
    )


class _AttentionSpy:
    """Records the ``attention_mask`` each cross-attention module receives."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[int, mx.array | None]] = []
        original = Attention.__call__
        spy = self

        def call(module, x, *args, **kwargs):
            if kwargs.get("encoder_hidden_states") is not None:
                spy.calls.append((id(module), kwargs.get("attention_mask")))
            return original(module, x, *args, **kwargs)

        monkeypatch.setattr(Attention, "__call__", call)

    def masks_for(self, module) -> list:
        return [mask for module_id, mask in self.calls if module_id == id(module)]


def test_block_routes_the_audio_mask_to_audio_attn2_only(monkeypatch):
    mx.random.seed(0)
    block = _block()
    kw = _block_kwargs(nv=8, na=6, nt=5)
    video_mask = mx.zeros((1, 1, 8, 5))
    audio_mask = mx.zeros((1, 1, 6, 5)).at[:, :, :, 2:].add(-1e4)
    spy = _AttentionSpy(monkeypatch)

    block(**kw, video_cross_attention_mask=video_mask, audio_cross_attention_mask=audio_mask)
    assert spy.masks_for(block.attn2) == [video_mask]
    assert spy.masks_for(block.audio_attn2) == [audio_mask]

    spy.calls.clear()
    block(**kw, video_cross_attention_mask=video_mask)
    assert spy.masks_for(block.attn2) == [video_mask]
    assert spy.masks_for(block.audio_attn2) == [None]


def test_block_audio_mask_changes_the_audio_output_and_none_is_a_no_op():
    mx.random.seed(1)
    block = _block()
    kw = _block_kwargs(nv=8, na=6, nt=5)
    v_ref, a_ref = block(**kw)
    v_none, a_none = block(**kw, audio_cross_attention_mask=None)
    assert mx.array_equal(v_ref, v_none).item() and mx.array_equal(a_ref, a_none).item()

    mask = mx.zeros((1, 1, 6, 5)).at[:, :, :, 2:].add(-1e4)
    _, a_masked = block(**kw, audio_cross_attention_mask=mask)
    assert float(mx.max(mx.abs(a_masked - a_ref))) > 1e-4


def _tiny_model() -> LTXModel:
    mx.random.seed(2)
    cfg = LTXModelConfig(
        num_layers=2,
        video_dim=64,
        audio_dim=32,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=16,
        audio_head_dim=8,
        av_cross_num_heads=4,
        av_cross_head_dim=8,
        video_patch_channels=16,
        audio_patch_channels=16,
        ff_mult=2.0,
        timestep_embedding_dim=32,
    )
    model = LTXModel(cfg)
    for block in model.transformer_blocks:
        for name in ("scale_shift_table", "audio_scale_shift_table"):
            block[name] = 0.5 * mx.random.normal(block[name].shape)
    mx.eval(model.parameters())
    return model


def _model_inputs(nv: int = 12, na: int = 6, nt: int = 5) -> dict:
    mx.random.seed(3)
    return dict(
        video_latent=mx.random.normal((1, nv, 16)),
        audio_latent=mx.random.normal((1, na, 16)),
        timestep=mx.array([0.7]),
        video_text_embeds=mx.random.normal((1, nt, 64)),
        audio_text_embeds=mx.random.normal((1, nt, 32)),
    )


def test_model_threads_the_audio_mask_to_every_block(monkeypatch):
    model = _tiny_model()
    mask = mx.zeros((1, 1, 6, 5)).at[:, :, :, 3:].add(-1e4)
    spy = _AttentionSpy(monkeypatch)
    model(**_model_inputs(), audio_cross_attention_mask=mask)
    for block in model.transformer_blocks:
        assert spy.masks_for(block.audio_attn2) == [mask]
        assert spy.masks_for(block.attn2) == [None]


def test_model_default_is_identical_with_the_kwarg_none():
    model = _tiny_model()
    v_ref, a_ref = model(**_model_inputs())
    v_none, a_none = model(**_model_inputs(), audio_cross_attention_mask=None)
    assert mx.array_equal(v_ref, v_none).item() and mx.array_equal(a_ref, a_none).item()
    mask = mx.zeros((1, 1, 6, 5)).at[:, :, :, 3:].add(-1e4)
    _, a_masked = model(**_model_inputs(), audio_cross_attention_mask=mask)
    assert float(mx.max(mx.abs(a_masked - a_ref))) > 1e-4


# --- sampler loops -------------------------------------------------------------------------------

B, NV, NA, NT, C = 1, 4, 3, 6, 8
_VMASK = mx.zeros((1, 1, NV, NT))
_AMASK = mx.full((1, 1, NA, NT), -1.0)


def _states(audio_frozen: bool = False) -> tuple[LatentState, LatentState]:
    video = LatentState(mx.zeros((B, NV, C)), mx.zeros((B, NV, C)), mx.ones((B, NV, 1)))
    if audio_frozen:
        audio = LatentState(mx.zeros((B, NA, C)), mx.zeros((B, NA, C)), mx.zeros((B, NA, 1)), frozen=True)
    else:
        audio = LatentState(mx.zeros((B, NA, C)), mx.zeros((B, NA, C)), mx.ones((B, NA, 1)))
    return video, audio


class _Recorder:
    """Model stand-in: records (video mask, audio mask, is-negative-pass) per call."""

    def __init__(self, negative: mx.array | None = None):
        self.calls: list[tuple] = []
        self.negative = negative

    def __call__(self, **kw):
        is_neg = self.negative is not None and kw["video_text_embeds"] is self.negative
        self.calls.append((kw.get("video_cross_attention_mask"), kw.get("audio_cross_attention_mask"), is_neg))
        return kw["video_latent"], kw["audio_latent"]


def _guiders(negative: mx.array):
    from ltx_core_mlx.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory

    params = MultiModalGuiderParams(cfg_scale=2.0, stg_scale=0.0, modality_scale=1.0, stg_blocks=[])
    return (
        create_multimodal_guider_factory(params, negative_context=negative),
        create_multimodal_guider_factory(params, negative_context=mx.zeros((B, NT, C))),
    )


def _run_loop(name: str, model, video, audio, **masks):
    from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
    from ltx_pipelines_mlx.utils import samplers

    common = dict(
        video_state=video,
        audio_state=audio,
        video_text_embeds=mx.zeros((B, NT, C)),
        audio_text_embeds=mx.zeros((B, NT, C)),
        sigmas=[1.0, 0.5, 0.0],
        show_progress=False,
        **masks,
    )
    if name == "euler":
        return samplers.denoise_loop(model=model, **common)
    if name == "ancestral":
        stepper = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)
        return samplers.euler_ancestral_denoising_loop(transformer=model, stepper=stepper, noise_seed=0, **common)
    vf, af = _guiders(model.negative)
    if name == "guided":
        return samplers.guided_denoise_loop(model=model, video_guider_factory=vf, audio_guider_factory=af, **common)
    if name == "res2s":
        return samplers.res2s_denoise_loop(model=model, video_guider_factory=vf, audio_guider_factory=af, **common)
    raise AssertionError(name)


_LOOPS = ["euler", "ancestral", "guided", "res2s"]


@pytest.mark.parametrize("loop", _LOOPS)
def test_loops_pass_the_audio_mask_on_positive_passes_only(loop):
    model = _Recorder(negative=mx.ones((B, NT, C)))
    _run_loop(loop, model, *_states(), video_cross_attention_mask=_VMASK, audio_cross_attention_mask=_AMASK)
    positive = [c for c in model.calls if not c[2]]
    negative = [c for c in model.calls if c[2]]
    assert positive and all(v is _VMASK and a is _AMASK for v, a, _ in positive)
    if loop in ("guided", "res2s"):
        assert negative and all(v is None and a is None for v, a, _ in negative)


@pytest.mark.parametrize("loop", _LOOPS)
def test_loops_drop_the_audio_mask_on_a_frozen_audio_stream(loop):
    model = _Recorder(negative=mx.ones((B, NT, C)))
    _run_loop(
        loop, model, *_states(audio_frozen=True), video_cross_attention_mask=_VMASK, audio_cross_attention_mask=_AMASK
    )
    assert model.calls and all(a is None for _, a, _ in model.calls)
    assert any(v is _VMASK for v, _, _ in model.calls)  # the video relay still applies


@pytest.mark.parametrize("loop", _LOOPS)
def test_loops_without_a_mask_never_pass_one(loop):
    model = _Recorder(negative=mx.ones((B, NT, C)))
    _run_loop(loop, model, *_states())
    assert model.calls and all(a is None and v is None for v, a, _ in model.calls)


def test_euler_loop_ignores_the_audio_mask_on_a_video_only_run():
    from ltx_pipelines_mlx.utils.samplers import denoise_loop

    seen: list = []

    def model(**kw):
        seen.append(kw.get("audio_cross_attention_mask"))
        return kw["video_latent"], None

    video, _ = _states()
    denoise_loop(
        model=model,
        video_state=video,
        audio_state=None,
        video_text_embeds=mx.zeros((B, NT, C)),
        audio_text_embeds=None,
        sigmas=[1.0, 0.0],
        audio_cross_attention_mask=_AMASK,
        show_progress=False,
    )
    assert seen == [None]


# --- pipelines: both masks reach every stage -----------------------------------------------------


class _Vae:
    def denormalize_latent(self, x):
        return x

    def normalize_latent(self, x):
        return x


class _EchoLoop:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

        self.calls.append(kwargs)
        return DenoiseOutput(video_latent=kwargs["video_state"].latent, audio_latent=kwargs["audio_state"].latent)


def _pack(tmp_path, ltx25: bool) -> str:
    transformer: dict = {"num_layers": 48}
    if ltx25:
        transformer["ff_bias"] = False
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))
    return str(tmp_path)


_RANGES = [(2, 4), (4, 6)]


def _stub_common(pipe, monkeypatch, mod) -> None:
    pipe.load = lambda: None
    pipe.dit = object()
    pipe.vae_encoder = _Vae()
    pipe.upsampler = lambda x: mx.repeat(mx.repeat(x, 2, axis=3), 2, axis=4)
    pipe._prompt_relay_setup = lambda prompt, relay: (prompt, list(_RANGES) if relay is not None else None)
    embeds = (mx.zeros((1, 8, 4096), dtype=mx.bfloat16), mx.zeros((1, 8, 2048), dtype=mx.bfloat16))
    pipe._encode_text = lambda prompt: embeds
    pipe._encode_text_with_negative = lambda prompt, negative_prompt=None: embeds + embeds
    pipe._load_text_encoder = lambda: None
    pipe._fuse_distilled_lora = lambda dit: None
    monkeypatch.setattr(mod, "X0Model", lambda dit: dit)


def _expected_audio_mask(relay, latent_f: int, num_audio: int) -> mx.array:
    lengths = distribute_segment_lengths(len(relay.local_prompts), latent_f, relay.segment_lengths)
    return build_relay_mask_audio(
        _RANGES, lengths, num_audio, latent_f, 8, epsilon=relay.epsilon, strength=relay.audio_strength
    )


def _pipeline_calls(name: str, tmp_path, monkeypatch, relay) -> list[dict]:
    """Run a stubbed ``generate_two_stage`` / ``generate_one_stage_dev``; return each loop call's kwargs."""
    from ltx_pipelines_mlx import distilled as dist_mod
    from ltx_pipelines_mlx import ti2vid_one_stage as one_mod
    from ltx_pipelines_mlx import ti2vid_two_stages as ts_mod
    from ltx_pipelines_mlx import ti2vid_two_stages_hq as hq_mod

    loop = _EchoLoop()
    kw = dict(prompt="a fox", height=128, width=128, num_frames=17, frame_rate=24.0, seed=7, prompt_relay=relay)
    if name in ("distilled-2.3", "distilled-2.5"):
        pipe = dist_mod.DistilledPipeline(model_dir=_pack(tmp_path, name.endswith("2.5")), low_memory=False)
        _stub_common(pipe, monkeypatch, dist_mod)
        monkeypatch.setattr(dist_mod, "denoise_loop", loop)
        monkeypatch.setattr(dist_mod, "euler_ancestral_denoising_loop", loop)
        pipe.generate_two_stage(**kw)
    elif name == "two-stage":
        pipe = ts_mod.TI2VidTwoStagesPipeline(model_dir=_pack(tmp_path, False))
        _stub_common(pipe, monkeypatch, ts_mod)
        monkeypatch.setattr(ts_mod, "guided_denoise_loop", loop)
        monkeypatch.setattr(ts_mod, "denoise_loop", loop)
        pipe.generate_two_stage(**kw)
    elif name == "two-stages-hq":
        pipe = hq_mod.TI2VidTwoStagesHQPipeline(model_dir=_pack(tmp_path, False))
        _stub_common(pipe, monkeypatch, hq_mod)
        monkeypatch.setattr(hq_mod, "res2s_denoise_loop", loop)
        pipe.generate_two_stage(**kw)
    else:
        pipe = one_mod.TI2VidOneStagePipeline(model_dir=_pack(tmp_path, False))
        _stub_common(pipe, monkeypatch, one_mod)
        monkeypatch.setattr(one_mod, "guided_denoise_loop", loop)
        pipe.generate_one_stage_dev(**kw, num_steps=2)
    return loop.calls


_PIPELINES = ["distilled-2.3", "distilled-2.5", "two-stage", "two-stages-hq", "one-stage"]


@pytest.mark.parametrize("name", _PIPELINES)
def test_pipelines_hand_every_stage_its_audio_mask(name, tmp_path, monkeypatch):
    relay = PromptRelayInput(["a red car", "a blue sky"], audio_strength=0.75)
    calls = _pipeline_calls(name, tmp_path, monkeypatch, relay)
    assert len(calls) == (1 if name == "one-stage" else 2)
    for call in calls:
        latent_f = 3  # 17 frames
        assert call["video_cross_attention_mask"] is not None
        if call["audio_state"].frozen:  # --two-stage stage 2: stage 1's audio is conditioning
            assert name == "two-stage" and call["audio_cross_attention_mask"] is None
            continue
        expected = _expected_audio_mask(relay, latent_f, call["audio_state"].latent.shape[1])
        assert mx.array_equal(call["audio_cross_attention_mask"], expected).item()


@pytest.mark.parametrize("name", _PIPELINES)
def test_pipelines_video_only_relay_passes_no_audio_mask(name, tmp_path, monkeypatch):
    calls = _pipeline_calls(name, tmp_path, monkeypatch, PromptRelayInput(["a", "b"], audio_strength=0.0))
    assert all(c["audio_cross_attention_mask"] is None and c["video_cross_attention_mask"] is not None for c in calls)
    calls = _pipeline_calls(name, tmp_path, monkeypatch, None)
    assert all(c["audio_cross_attention_mask"] is None and c["video_cross_attention_mask"] is None for c in calls)


# --- CLI -----------------------------------------------------------------------------------------


def _generate_args(tmp_path, *extra: str):
    from ltx_pipelines_mlx.cli import _build_parser

    argv = ["generate", "-p", "x", "-o", "o.mp4", "--frame-rate", "24", "-f", "17", "--model", _pack(tmp_path, True)]
    return _build_parser().parse_args([*argv, "--distilled", *extra])


def _captured_relay(tmp_path, monkeypatch, *extra: str):
    from ltx_pipelines_mlx import distilled as dist_mod
    from ltx_pipelines_mlx.cli import _cmd_generate

    captured: dict = {}

    class _FakePipe:
        def __init__(self, **kw):
            pass

        def generate_and_save(self, **kw):
            captured.update(kw)
            return "o.mp4"

    monkeypatch.setattr(dist_mod, "DistilledPipeline", _FakePipe)
    _cmd_generate(_generate_args(tmp_path, "-q", *extra))
    return captured.get("prompt_relay")


def test_cli_audio_flags_reach_the_relay(tmp_path, monkeypatch):
    relay = _captured_relay(tmp_path, monkeypatch, "--segment", "a", "--segment", "b")
    assert (relay.audio_strength, relay.audio_epsilon) == (None, None)  # the pipeline's default
    relay = _captured_relay(
        tmp_path,
        monkeypatch,
        "--segment",
        "a",
        "--segment",
        "b",
        "--relay-audio-strength",
        "0",
        "--relay-audio-epsilon",
        "0.01",
    )
    assert (relay.audio_strength, relay.audio_epsilon) == (0.0, 0.01)


def test_cli_rejects_a_bad_audio_epsilon_before_building_the_pipeline(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="epsilon"):
        _captured_relay(tmp_path, monkeypatch, "--segment", "a", "--relay-audio-epsilon", "2")


def test_builder_default_strength_applies_only_when_none_is_given():
    assert _audio_builder(PromptRelayInput(["a", "b"]), default=0.0)(13, _audio()) is None
    on = _audio_builder(PromptRelayInput(["a", "b"]), default=1.0)(13, _audio())
    explicit = _audio_builder(PromptRelayInput(["a", "b"], audio_strength=1.0), default=0.0)(13, _audio())
    assert on is not None and mx.array_equal(on, explicit).item()


@pytest.mark.parametrize("name", _PIPELINES)
def test_default_audio_relay_is_on_for_distilled_and_off_for_cfg(name, tmp_path, monkeypatch):
    calls = _pipeline_calls(name, tmp_path, monkeypatch, PromptRelayInput(["a red car", "a blue sky"]))
    generated = [c for c in calls if not c["audio_state"].frozen]
    assert generated
    if name.startswith("distilled"):
        relay = PromptRelayInput(["a red car", "a blue sky"], audio_strength=1.0)
        for call in generated:
            expected = _expected_audio_mask(relay, 3, call["audio_state"].latent.shape[1])
            assert mx.array_equal(call["audio_cross_attention_mask"], expected).item()
    else:
        assert all(c["audio_cross_attention_mask"] is None for c in calls)
        assert all(c["video_cross_attention_mask"] is not None for c in calls)
