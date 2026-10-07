"""Normalized Attention Guidance (NAG): a negative prompt on the CFG-less distilled / DFR paths.

The reference is kijai/ComfyUI-KJNodes ``LTX2_NAG`` (``nodes/ltxv_nodes.py``) and the NAG paper's
attention processors (ChenDarYen/Normalized-Attention-Guidance). These tests pin the guidance math
against a numpy transcription of the node, its place inside ``Attention`` (unmasked negative,
before the gate), the per-step prompt AdaLN on the negative context, its path from the CLI and the
pipelines through the two distilled loops and the model to every ``attn2`` / ``audio_attn2``, and
the cases where nothing may change: NAG off, ``scale == 1``, a frozen or absent audio stream.
Pure CPU, tiny modules, no weights.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import ltx_pipelines_mlx.distilled as distilled_mod
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.guidance.nag import (
    DEFAULT_NAG_ALPHA,
    DEFAULT_NAG_SCALE,
    DEFAULT_NAG_TAU,
    NAGConfig,
    NAGGuidance,
    normalized_attention_guidance,
)
from ltx_core_mlx.loader.block_streaming import BlockStreamer, StreamingLTXModel
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock
from ltx_pipelines_mlx.cli import _build_parser, _cmd_generate
from tests.test_compute_dtype import _inputs, _model, _rel_err, _save_blocks

# --- the guidance math ---------------------------------------------------------------------------


def _kijai_nag(pos: np.ndarray, neg: np.ndarray, scale: float, alpha: float, tau: float) -> np.ndarray:
    """``normalized_attention_guidance`` of kijai's LTX2_NAG (non-inplace branch), in numpy float64."""
    guidance = pos * scale - neg * (scale - 1)
    norm_pos = np.abs(pos).sum(axis=-1, keepdims=True)
    norm_guidance = np.abs(guidance).sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.nan_to_num(norm_guidance / norm_pos, nan=10.0)
    adjustment = (norm_pos * tau) / (norm_guidance + 1e-7)
    guidance = guidance * np.where(ratio > tau, adjustment, 1.0)
    return guidance * alpha + pos * (1 - alpha)


def _pair(seed: int = 0, shape=(2, 7, 64), spread: float = 1.0) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=shape).astype(np.float32)
    # A negative close to the positive for some tokens (small ratio, no clip) and far for others.
    neg = pos + spread * rng.normal(size=shape).astype(np.float32) * rng.uniform(0, 1, size=shape[:-1] + (1,))
    return mx.array(pos), mx.array(neg.astype(np.float32))


@pytest.mark.parametrize(("scale", "alpha", "tau"), [(11.0, 0.25, 2.5), (5.0, 0.5, 1.5), (3.0, 1.0, 4.0)])
def test_math_matches_the_reference_node(scale, alpha, tau):
    pos, neg = _pair()
    out = normalized_attention_guidance(pos, neg, scale, alpha, tau)
    expected = _kijai_nag(np.array(pos, np.float64), np.array(neg, np.float64), scale, alpha, tau)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-5, atol=1e-5)


def test_tau_clips_some_tokens_and_not_others():
    pos, neg = _pair(spread=0.5)
    scale, tau = 11.0, 2.5
    guidance = np.array(pos) * scale - np.array(neg) * (scale - 1)
    ratio = np.abs(guidance).sum(-1) / np.abs(np.array(pos)).sum(-1)
    assert (ratio > tau).any() and (ratio <= tau).any()  # the test data exercises both branches
    guided = np.array(normalized_attention_guidance(pos, neg, scale, 1.0, tau))  # alpha 1: the guided part
    l1 = np.abs(guided).sum(-1)
    l1_pos = np.abs(np.array(pos)).sum(-1)
    np.testing.assert_allclose(l1[ratio > tau], tau * l1_pos[ratio > tau], rtol=1e-4)  # clipped to tau
    np.testing.assert_allclose(guided[ratio <= tau], guidance[ratio <= tau], rtol=1e-5, atol=1e-5)  # untouched


def test_alpha_blends_with_the_positive():
    pos, neg = _pair()
    np.testing.assert_array_equal(np.array(normalized_attention_guidance(pos, neg, 11.0, 0.0, 2.5)), np.array(pos))
    full = normalized_attention_guidance(pos, neg, 11.0, 1.0, 2.5)
    half = normalized_attention_guidance(pos, neg, 11.0, 0.5, 2.5)
    np.testing.assert_allclose(np.array(half), 0.5 * np.array(full) + 0.5 * np.array(pos), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("case", ["scale_1", "negative_equals_positive"])
def test_identity_cases(case):
    pos, neg = _pair()
    if case == "scale_1":
        out = normalized_attention_guidance(pos, neg, 1.0, DEFAULT_NAG_ALPHA, DEFAULT_NAG_TAU)
    else:
        out = normalized_attention_guidance(pos, pos, DEFAULT_NAG_SCALE, DEFAULT_NAG_ALPHA, DEFAULT_NAG_TAU)
    np.testing.assert_allclose(np.array(out), np.array(pos), rtol=1e-5, atol=1e-5)


def test_head_layout_matches_the_flattened_reference():
    """Inside Attention the outputs are (B, H, N, D); the norm must span every head of a token."""
    pos, neg = _pair(shape=(1, 4, 9, 16))  # (B, H, N, D)
    out = normalized_attention_guidance(pos, neg, 11.0, 0.25, 2.5, axis=(1, 3))
    flat = lambda x: np.array(x).transpose(0, 2, 1, 3).reshape(1, 9, 64)  # noqa: E731
    expected = _kijai_nag(flat(pos).astype(np.float64), flat(neg).astype(np.float64), 11.0, 0.25, 2.5)
    np.testing.assert_allclose(flat(out), expected, rtol=1e-5, atol=1e-5)


def test_zero_tokens_stay_finite_and_match_the_reference():
    pos, neg = _pair()
    pos = pos.at[0, 3].multiply(0.0)  # zero positive, nonzero negative: ratio +inf, clipped to tau * 0
    pos, neg = pos.at[1, 2].multiply(0.0), neg.at[1, 2].multiply(0.0)  # both zero: ratio 0/0
    out = normalized_attention_guidance(pos, neg, 11.0, 0.25, 2.5)
    assert bool(mx.all(mx.isfinite(out)).item())
    np.testing.assert_allclose(np.array(out[0, 3]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.array(out[1, 2]), 0.0, atol=1e-6)
    expected = _kijai_nag(np.array(pos, np.float64), np.array(neg, np.float64), 11.0, 0.25, 2.5)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-5, atol=1e-5)


def test_float16_inputs_are_combined_in_float32():
    """11 x the attention output and an L1 sum over 4,096 features both leave float16's range."""
    rng = np.random.default_rng(1)
    pos = mx.array(rng.normal(scale=2000.0, size=(1, 4, 4096)).astype(np.float16))
    neg = mx.array(rng.normal(scale=2000.0, size=(1, 4, 4096)).astype(np.float16))
    naive = pos * 11.0 - neg * 10.0  # what float16 arithmetic would do
    assert not bool(mx.all(mx.isfinite(mx.sum(mx.abs(naive), axis=-1))).item())
    out = normalized_attention_guidance(pos, neg, 11.0, 0.25, 2.5)
    assert out.dtype == mx.float16 and bool(mx.all(mx.isfinite(out)).item())
    expected = _kijai_nag(np.array(pos, np.float64), np.array(neg, np.float64), 11.0, 0.25, 2.5)
    np.testing.assert_allclose(np.array(out, np.float64), expected, rtol=2e-3, atol=2.0)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"scale": 0.5}, "scale"),
        ({"scale": float("nan")}, "scale"),
        ({"alpha": 1.5}, "alpha"),
        ({"alpha": -0.1}, "alpha"),
        ({"tau": 0.0}, "tau"),
    ],
)
def test_config_validates_up_front(kw, match):
    with pytest.raises(ValueError, match=match):
        NAGConfig(**kw)


def test_config_defaults_and_activity():
    cfg = NAGConfig()
    assert (cfg.scale, cfg.alpha, cfg.tau, cfg.audio) == (11.0, 0.25, 2.5, True)  # kijai's LTX2_NAG defaults
    assert cfg.active and not NAGConfig(scale=1.0).active


def test_guidance_drops_the_audio_context_when_audio_is_off():
    v, a = mx.zeros((1, 3, 8)), mx.zeros((1, 3, 4))
    assert NAGGuidance.from_config(NAGConfig(), v, a).audio_text_embeds is a
    assert NAGGuidance.from_config(NAGConfig(audio=False), v, a).audio_text_embeds is None


# --- Attention -----------------------------------------------------------------------------------


def _attention(gated: bool = True) -> Attention:
    mx.random.seed(4)
    attn = Attention(query_dim=32, kv_dim=24, num_heads=4, head_dim=8, use_rope=False, apply_gated_attention=gated)
    if gated:
        attn.to_gate_logits.weight = 0.3 * mx.random.normal(attn.to_gate_logits.weight.shape)
    mx.eval(attn.parameters())
    return attn


def _reference_attention(attn, x, ctx, neg, mask, scale, alpha, tau) -> np.ndarray:
    """kijai's patched forward: q once, positive (masked) and negative (unmasked) attention, NAG, gate, to_out."""
    b, n, _ = x.shape
    q = attn.q_norm(attn.to_q(x)).reshape(b, n, 4, 8).transpose(0, 2, 1, 3)

    def attend(c, m):
        k = attn.k_norm(attn.to_k(c)).reshape(b, -1, 4, 8).transpose(0, 2, 1, 3)
        v = attn.to_v(c).reshape(b, -1, 4, 8).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=m)
        return np.array(o.transpose(0, 2, 1, 3).reshape(b, n, 32), np.float64)

    z = _kijai_nag(attend(ctx, mask), attend(neg, None), scale, alpha, tau)
    if attn.to_gate_logits is not None:
        gate = 2.0 / (1.0 + np.exp(-np.array(attn.to_gate_logits(x), np.float64)))  # (B, N, H)
        z = (z.reshape(b, n, 4, 8) * gate[..., None]).reshape(b, n, 32)
    return np.array(attn.to_out(mx.array(z.astype(np.float32))))


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_attention_matches_the_reference_forward(gated, masked):
    attn = _attention(gated)
    x, ctx, neg = mx.random.normal((1, 6, 32)), mx.random.normal((1, 5, 24)), mx.random.normal((1, 5, 24))
    mask = mx.zeros((1, 1, 6, 5)).at[:, :, :3, 2:].add(-1e4) if masked else None
    out = attn(x, encoder_hidden_states=ctx, attention_mask=mask, nag_encoder_hidden_states=neg)
    expected = _reference_attention(attn, x, ctx, neg, mask, DEFAULT_NAG_SCALE, DEFAULT_NAG_ALPHA, DEFAULT_NAG_TAU)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-4, atol=1e-4)


def test_attention_mask_never_reaches_the_negative(monkeypatch):
    attn = _attention()
    seen = []
    real = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, scale, mask=None):
        seen.append(mask)
        return real(q, k, v, scale=scale, mask=mask)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    mask = mx.zeros((1, 1, 6, 5))
    attn(mx.ones((1, 6, 32)), encoder_hidden_states=mx.ones((1, 5, 24)), attention_mask=mask,
         nag_encoder_hidden_states=mx.zeros((1, 5, 24)))  # fmt: skip
    assert len(seen) == 2 and seen[0] is not None and seen[1] is None


def test_attention_without_a_negative_is_unchanged():
    attn = _attention()
    x, ctx = mx.random.normal((1, 6, 32)), mx.random.normal((1, 5, 24))
    assert mx.array_equal(attn(x, encoder_hidden_states=ctx), attn(x, encoder_hidden_states=ctx)).item()
    ref = attn(x, encoder_hidden_states=ctx)
    assert mx.array_equal(attn(x, encoder_hidden_states=ctx, nag_encoder_hidden_states=None), ref).item()


def test_attention_refuses_nag_on_self_attention():
    with pytest.raises(ValueError, match="cross-attention"):
        _attention()(mx.ones((1, 6, 32)), nag_encoder_hidden_states=mx.ones((1, 5, 24)))


def test_attention_float16_compute_keeps_both_kernels_in_float16(monkeypatch):
    from ltx_core_mlx.model.transformer.transformer import cast_float_params

    attn = _attention()
    x, ctx, neg = (mx.random.normal((1, 6, 32)), mx.random.normal((1, 5, 24)), mx.random.normal((1, 5, 24)))
    ref = attn(x, encoder_hidden_states=ctx, nag_encoder_hidden_states=neg)
    attn.compute_dtype = mx.float16  # what LTXModel.set_compute_dtype does per module
    attn.update(cast_float_params(attn.parameters(), mx.float16))
    dtypes = []
    real = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, scale, mask=None):
        dtypes.append((q.dtype, k.dtype, v.dtype))
        return real(q, k, v, scale=scale, mask=mask)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    out = attn(x, encoder_hidden_states=ctx, nag_encoder_hidden_states=neg)
    assert dtypes == [(mx.float16,) * 3] * 2  # positive and negative attention
    assert out.dtype == mx.float16 and bool(mx.all(mx.isfinite(out)).item())
    assert _rel_err(out, ref) < 5e-3


# --- block and model -----------------------------------------------------------------------------


def _block() -> BasicAVTransformerBlock:
    mx.random.seed(5)
    block = BasicAVTransformerBlock(
        video_dim=32,
        audio_dim=16,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=4,
        av_cross_head_dim=4,
    )
    block.prompt_scale_shift_table = 0.3 * mx.random.normal((2, 32))
    block.audio_prompt_scale_shift_table = 0.3 * mx.random.normal((2, 16))
    return block


def _block_kwargs(nv: int = 8, na: int = 6, nt: int = 5) -> dict:
    mx.random.seed(6)
    video_adaln = mx.zeros((1, 9 * 32)).at[:, 8 * 32 :].add(1.0)  # nonzero text cross-attention gates
    audio_adaln = mx.zeros((1, 9 * 16)).at[:, 8 * 16 :].add(1.0)
    return dict(
        video_hidden=mx.random.normal((1, nv, 32)),
        audio_hidden=mx.random.normal((1, na, 16)),
        video_adaln_params=video_adaln,
        audio_adaln_params=audio_adaln,
        video_prompt_adaln_params=0.2 * mx.random.normal((1, 2 * 32)),
        audio_prompt_adaln_params=0.2 * mx.random.normal((1, 2 * 16)),
        av_ca_video_params=mx.zeros((1, 4 * 32)),
        av_ca_audio_params=mx.zeros((1, 4 * 16)),
        av_ca_a2v_gate_params=mx.zeros((1, 32)),
        av_ca_v2a_gate_params=mx.zeros((1, 16)),
        video_text_embeds=mx.random.normal((1, nt, 32)),
        audio_text_embeds=mx.random.normal((1, nt, 16)),
    )


class _AttentionSpy:
    """Records the keyword arguments of every cross-attention call."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[int, dict]] = []
        original = Attention.__call__
        spy = self

        def call(module, x, *args, **kwargs):
            if kwargs.get("encoder_hidden_states") is not None:
                spy.calls.append((id(module), dict(kwargs)))
            return original(module, x, *args, **kwargs)

        monkeypatch.setattr(Attention, "__call__", call)

    def for_module(self, module) -> list[dict]:
        return [kw for module_id, kw in self.calls if module_id == id(module)]


def _negatives(nt: int = 5) -> tuple[mx.array, mx.array]:
    mx.random.seed(7)
    return mx.random.normal((1, nt, 32)), mx.random.normal((1, nt, 16))


def test_block_modulates_the_negative_like_the_positive(monkeypatch):
    block = _block()
    kw = _block_kwargs()
    neg_v, neg_a = _negatives()
    nag = NAGGuidance(neg_v, neg_a, scale=11.0, alpha=0.25, tau=2.5)
    spy = _AttentionSpy(monkeypatch)
    block(**kw, nag=nag)

    (video_call,) = spy.for_module(block.attn2)
    (audio_call,) = spy.for_module(block.audio_attn2)
    vp = kw["video_prompt_adaln_params"].reshape(1, 2, 32) + block.prompt_scale_shift_table[None]
    ap = kw["audio_prompt_adaln_params"].reshape(1, 2, 16) + block.audio_prompt_scale_shift_table[None]
    for call, neg, pos, p in ((video_call, neg_v, kw["video_text_embeds"], vp), (audio_call, neg_a, kw["audio_text_embeds"], ap)):  # fmt: skip
        shift, scale = p[:, 0:1], p[:, 1:2]
        assert mx.allclose(call["encoder_hidden_states"], pos * (1 + scale) + shift).item()
        assert mx.allclose(call["nag_encoder_hidden_states"], neg * (1 + scale) + shift).item()
        assert (call["nag_scale"], call["nag_alpha"], call["nag_tau"]) == (11.0, 0.25, 2.5)


def test_block_without_an_audio_negative_leaves_audio_attn2_unguided(monkeypatch):
    block = _block()
    neg_v, _ = _negatives()
    spy = _AttentionSpy(monkeypatch)
    block(**_block_kwargs(), nag=NAGGuidance(neg_v, None, 11.0, 0.25, 2.5))
    assert "nag_encoder_hidden_states" in spy.for_module(block.attn2)[0]
    assert "nag_encoder_hidden_states" not in spy.for_module(block.audio_attn2)[0]


def test_block_nag_changes_the_output_and_none_is_a_no_op():
    block = _block()
    kw = _block_kwargs()
    v_ref, a_ref = block(**kw)
    v_none, a_none = block(**kw, nag=None)
    assert mx.array_equal(v_ref, v_none).item() and mx.array_equal(a_ref, a_none).item()
    neg_v, neg_a = _negatives()
    v_nag, a_nag = block(**kw, nag=NAGGuidance(neg_v, neg_a, 11.0, 0.25, 2.5))
    assert float(mx.max(mx.abs(v_nag - v_ref))) > 1e-3 and float(mx.max(mx.abs(a_nag - a_ref))) > 1e-3


def _model_nag(cfg, *, scale: float = 11.0, audio: bool = True, same_as: dict | None = None) -> NAGGuidance:
    mx.random.seed(8)
    if same_as is not None:
        return NAGGuidance(same_as["video_text_embeds"], same_as["audio_text_embeds"], scale, 0.25, 2.5)
    neg_v = mx.random.normal((1, 6, cfg.video_dim)).astype(mx.bfloat16)
    neg_a = mx.random.normal((1, 6, cfg.audio_dim)).astype(mx.bfloat16)
    return NAGGuidance(neg_v, neg_a if audio else None, scale, 0.25, 2.5)


def test_model_threads_nag_to_every_block(monkeypatch):
    model = _model()
    spy = _AttentionSpy(monkeypatch)
    model(**_inputs(model.config), nag=_model_nag(model.config))
    for block in model.transformer_blocks:
        for module in (block.attn2, block.audio_attn2):
            (call,) = spy.for_module(module)
            assert call["nag_encoder_hidden_states"] is not None


def test_model_default_is_identical_and_nag_changes_both_streams():
    model = _model()
    inputs = _inputs(model.config)
    v_ref, a_ref = model(**inputs)
    v_none, a_none = model(**inputs, nag=None)
    assert mx.array_equal(v_ref, v_none).item() and mx.array_equal(a_ref, a_none).item()
    v, a = model(**inputs, nag=_model_nag(model.config))
    assert _rel_err(v, v_ref) > 1e-3 and _rel_err(a, a_ref) > 1e-3
    # scale 1 and negative == positive are the identity (up to float rounding of the blend)
    for nag in (_model_nag(model.config, scale=1.0), _model_nag(model.config, same_as=inputs)):
        v1, a1 = model(**inputs, nag=nag)
        assert _rel_err(v1, v_ref) < 1e-3 and _rel_err(a1, a_ref) < 1e-3


def test_model_video_only_path_ignores_the_audio_negative():
    model = _model()
    inputs = {**_inputs(model.config), "audio_latent": None, "audio_text_embeds": None}
    v_audio_neg, a = model(**inputs, nag=_model_nag(model.config))
    v_no_audio_neg, _ = model(**inputs, nag=_model_nag(model.config, audio=False))
    assert a is None and mx.array_equal(v_audio_neg, v_no_audio_neg).item()


def test_model_float16_compute_with_nag_is_finite_and_close():
    ref_model, model = _model(), _model()
    model.set_compute_dtype(mx.float16)
    inputs = _inputs(ref_model.config)
    ref_v, ref_a = ref_model(**inputs, nag=_model_nag(ref_model.config))
    v, a = model(**inputs, nag=_model_nag(model.config))
    assert model.compute_dtype == mx.float16  # no overflow fallback
    assert _rel_err(v, ref_v) < 2e-2 and _rel_err(a, ref_a) < 2e-2


def test_streamed_compiled_block_matches_resident_with_nag():
    """--low-ram compiles the shared block: NAGGuidance (a NamedTuple) must cross mx.compile."""
    resident = _model()
    inputs = _inputs(resident.config)
    nag = _model_nag(resident.config)
    ref_v, ref_a = resident(**inputs, nag=nag)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "blocks.safetensors"
        _save_blocks(_model(), path)
        inner = _model()
        inner.transformer_blocks = [inner.transformer_blocks[0]]
        streamed = StreamingLTXModel(inner, BlockStreamer(path, block_prefix="transformer_blocks."))
        v, a = streamed(**inputs, nag=nag)
        assert _rel_err(v, ref_v) < 1e-5 and _rel_err(a, ref_a) < 1e-5
        v0, _ = streamed(**inputs)  # and back to plain: the compiled block retraces
        assert _rel_err(v0, resident(**inputs)[0]) < 1e-5


# --- sampler loops -------------------------------------------------------------------------------

B, NV, NA, NT, C = 1, 4, 3, 6, 8
_NAG = NAGGuidance(mx.ones((B, NT, C)), mx.ones((B, NT, C)), 11.0, 0.25, 2.5)


def _states(audio_frozen: bool = False) -> tuple[LatentState, LatentState]:
    video = LatentState(mx.zeros((B, NV, C)), mx.zeros((B, NV, C)), mx.ones((B, NV, 1)))
    mask = mx.zeros((B, NA, 1)) if audio_frozen else mx.ones((B, NA, 1))
    audio = LatentState(mx.zeros((B, NA, C)), mx.zeros((B, NA, C)), mask, frozen=audio_frozen)
    return video, audio


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return kw["video_latent"], kw["audio_latent"]


def _run_loop(name: str, model, video, audio, **extra):
    from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
    from ltx_pipelines_mlx.utils import samplers

    common = dict(
        video_state=video,
        audio_state=audio,
        video_text_embeds=mx.zeros((B, NT, C)),
        audio_text_embeds=mx.zeros((B, NT, C)),
        sigmas=[1.0, 0.5, 0.0],
        show_progress=False,
        **extra,
    )
    if name == "euler":
        return samplers.denoise_loop(model=model, **common)
    stepper = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)
    return samplers.euler_ancestral_denoising_loop(transformer=model, stepper=stepper, noise_seed=0, **common)


@pytest.mark.parametrize("loop", ["euler", "ancestral"])
def test_loops_pass_nag_to_every_forward(loop):
    model = _Recorder()
    _run_loop(loop, model, *_states(), nag=_NAG)
    assert len(model.calls) == 2 and all(c["nag"] is _NAG for c in model.calls)


@pytest.mark.parametrize("loop", ["euler", "ancestral"])
def test_loops_without_nag_never_pass_the_kwarg(loop):
    model = _Recorder()
    _run_loop(loop, model, *_states())
    assert model.calls and all("nag" not in c for c in model.calls)


@pytest.mark.parametrize("loop", ["euler", "ancestral"])
def test_loops_drop_the_audio_negative_on_a_frozen_audio_stream(loop):
    model = _Recorder()
    _run_loop(loop, model, *_states(audio_frozen=True), nag=_NAG)
    for call in model.calls:
        assert call["nag"].audio_text_embeds is None and call["nag"].video_text_embeds is _NAG.video_text_embeds


def test_euler_loop_drops_the_audio_negative_on_a_video_only_run():
    from ltx_pipelines_mlx.utils.samplers import denoise_loop

    seen: list = []

    def model(**kw):
        seen.append(kw["nag"])
        return kw["video_latent"], None

    video, _ = _states()
    denoise_loop(
        model=model,
        video_state=video,
        audio_state=None,
        video_text_embeds=mx.zeros((B, NT, C)),
        audio_text_embeds=None,
        sigmas=[1.0, 0.0],
        nag=_NAG,
        show_progress=False,
    )
    assert len(seen) == 1 and seen[0].audio_text_embeds is None


# --- pipelines -----------------------------------------------------------------------------------


def _pack(tmp_path, ltx25: bool) -> str:
    transformer: dict = {"num_layers": 48}
    if ltx25:
        transformer["ff_bias"] = False
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))
    return str(tmp_path)


def _distilled(tmp_path, monkeypatch, ltx25: bool):
    from tests.test_ltx25_distilled import _fake_upsampler, _FakeVaeEncoder, _LoopSpy

    pipe = distilled_mod.DistilledPipeline(model_dir=_pack(tmp_path, ltx25), low_memory=False)
    pipe.load = lambda: None
    pipe.dit = object()
    pipe.vae_encoder = _FakeVaeEncoder()
    pipe.upsampler = _fake_upsampler
    pipe._load_text_encoder = lambda: None
    encoded: list[tuple[str, tuple]] = []

    def encode(prompt):
        out = (mx.full((1, 8, 4096), len(encoded), dtype=mx.bfloat16), mx.zeros((1, 8, 2048), dtype=mx.bfloat16))
        encoded.append((prompt, out))
        return out

    pipe._encode_text = encode
    monkeypatch.setattr(distilled_mod, "X0Model", lambda dit: dit)
    loop = _LoopSpy()
    monkeypatch.setattr(distilled_mod, "denoise_loop", loop)
    monkeypatch.setattr(distilled_mod, "euler_ancestral_denoising_loop", loop)
    return pipe, loop, encoded


_GEN = dict(prompt="a fox", height=128, width=128, num_frames=17, frame_rate=24.0, seed=7)


@pytest.mark.parametrize("ltx25", [False, True])
def test_distilled_encodes_the_negative_once_and_guides_both_stages(tmp_path, monkeypatch, ltx25):
    pipe, loop, encoded = _distilled(tmp_path, monkeypatch, ltx25)
    cfg = NAGConfig(scale=7.0, alpha=0.5, tau=2.0)
    pipe.generate_two_stage(**_GEN, negative_prompt="mouth wide open", nag=cfg)
    assert [p for p, _ in encoded] == ["a fox", "mouth wide open"]
    neg_v, neg_a = encoded[1][1]
    assert len(loop.calls) == 2
    for call in loop.calls:
        nag = call["nag"]
        assert nag.video_text_embeds is neg_v and nag.audio_text_embeds is neg_a
        assert (nag.scale, nag.alpha, nag.tau) == (7.0, 0.5, 2.0)
        assert call["video_text_embeds"] is encoded[0][1][0]  # the positive is untouched


@pytest.mark.parametrize("ltx25", [False, True])
def test_distilled_without_nag_makes_the_same_calls_as_before(tmp_path, monkeypatch, ltx25):
    pipe, loop, encoded = _distilled(tmp_path, monkeypatch, ltx25)
    pipe.generate_two_stage(**_GEN)
    assert [p for p, _ in encoded] == ["a fox"]
    assert len(loop.calls) == 2 and all("nag" not in c for c in loop.calls)


def test_distilled_nag_scale_1_is_off(tmp_path, monkeypatch):
    pipe, loop, encoded = _distilled(tmp_path, monkeypatch, True)
    pipe.generate_two_stage(**_GEN, negative_prompt="blurry", nag=NAGConfig(scale=1.0))
    assert [p for p, _ in encoded] == ["a fox"] and all("nag" not in c for c in loop.calls)


def test_distilled_video_only_nag_config(tmp_path, monkeypatch):
    pipe, loop, _ = _distilled(tmp_path, monkeypatch, True)
    pipe.generate_two_stage(**_GEN, negative_prompt="blurry", nag=NAGConfig(audio=False))
    assert all(c["nag"].audio_text_embeds is None for c in loop.calls)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"negative_prompt": "blurry"}, "negative_prompt requires a CFG pipeline.*nag=NAGConfig"),
        ({"nag": NAGConfig()}, "nag needs a negative_prompt"),
    ],
)
def test_distilled_refuses_half_a_nag_request_before_encoding(tmp_path, monkeypatch, kw, match):
    pipe, _, encoded = _distilled(tmp_path, monkeypatch, True)
    with pytest.raises(ValueError, match=match):
        pipe.generate_two_stage(**_GEN, **kw)
    assert encoded == []


def test_nag_composes_with_prompt_relay(tmp_path, monkeypatch):
    from ltx_core_mlx.conditioning.prompt_relay import PromptRelayInput

    pipe, loop, encoded = _distilled(tmp_path, monkeypatch, True)
    pipe._prompt_relay_setup = lambda prompt, relay: ("a fox. runs. sleeps", [(2, 3), (3, 4)])
    relay = PromptRelayInput(["runs", "sleeps"])
    pipe.generate_two_stage(**_GEN, prompt_relay=relay, negative_prompt="blurry", nag=NAGConfig())
    assert [p for p, _ in encoded] == ["a fox. runs. sleeps", "blurry"]  # combined positive, global negative
    for call in loop.calls:
        assert call["video_cross_attention_mask"] is not None and call["nag"] is not None


def test_generate_and_save_forwards_nag(tmp_path, monkeypatch):
    pipe, _, _ = _distilled(tmp_path, monkeypatch, True)
    seen = {}

    def gen(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(pipe, "generate_two_stage", gen)
    cfg = NAGConfig()
    with pytest.raises(RuntimeError):
        pipe.generate_and_save(**_GEN, output_path="x.mp4", negative_prompt="n", nag=cfg)
    assert seen["nag"] is cfg and seen["negative_prompt"] == "n"
    seen.clear()
    with pytest.raises(RuntimeError):
        pipe.generate_and_save(**_GEN, output_path="x.mp4")
    assert "nag" not in seen and "negative_prompt" not in seen


def test_dfr_guides_every_pass_with_the_negative(tmp_path, monkeypatch):
    """Stage 1, stage 2, both temporal-round tiles and the four epilogue passes."""
    from tests.test_dfr import _run
    from tests.test_dfr_epilogue import _make_epilogue

    pipe, _, _, _, _, rec = _make_epilogue(tmp_path, monkeypatch, t=1)
    prompts = []
    real_encode = pipe._encode_text

    def encode(prompt):
        prompts.append(prompt)
        return real_encode(prompt)

    pipe._encode_text = encode
    _run(pipe, height=256, width=256, num_frames=49, negative_prompt="extra fingers", nag=NAGConfig())
    calls = rec["ancestral"].calls
    assert prompts == ["a fox", "extra fingers"]
    assert len(calls) == 2 + 2 + 4
    nags = {id(c["nag"]) for c in calls}
    assert len(nags) == 1 and calls[0]["nag"] is not None


def test_dfr_without_nag_passes_none_and_refuses_a_lone_negative(tmp_path, monkeypatch):
    from tests.test_dfr import _make, _run

    pipe, _, ancestral, _, _ = _make(tmp_path, monkeypatch)
    _run(pipe)
    assert ancestral.calls and all("nag" not in c for c in ancestral.calls)
    with pytest.raises(ValueError, match="negative_prompt requires a CFG pipeline"):
        _run(pipe, negative_prompt="n")


# --- CLI -----------------------------------------------------------------------------------------


def _argv(tmp_path, *extra: str) -> list[str]:
    pack = _pack(tmp_path, True)
    return ["generate", "-p", "x", "-o", "o.mp4", "--frame-rate", "24", "-f", "17", "--model", pack, "-q", *extra]


def _captured(tmp_path, monkeypatch, mode: str, *extra: str) -> dict:
    from ltx_pipelines_mlx import dfr as dfr_mod

    captured: dict = {}

    class _FakePipe:
        def __init__(self, **kw):
            pass

        def generate_and_save(self, **kw):
            captured.update(kw)
            return "o.mp4"

    monkeypatch.setattr(distilled_mod, "DistilledPipeline", _FakePipe)
    monkeypatch.setattr(dfr_mod, "DFRPipeline", _FakePipe)
    _cmd_generate(_build_parser().parse_args(_argv(tmp_path, mode, *extra)))
    return captured


@pytest.mark.parametrize("mode", ["--distilled", "--dfr"])
def test_cli_nag_reaches_the_pipeline_with_defaults(tmp_path, monkeypatch, mode):
    kw = _captured(tmp_path, monkeypatch, mode, "--nag", "--negative-prompt", "mouth wide open")
    assert kw["negative_prompt"] == "mouth wide open" and kw["nag"] == NAGConfig()


def test_cli_nag_tuning_flags(tmp_path, monkeypatch):
    kw = _captured(
        tmp_path,
        monkeypatch,
        "--distilled",
        "--nag",
        "--negative-prompt",
        "",
        "--nag-scale",
        "5",
        "--nag-alpha",
        "0.5",
        "--nag-tau",
        "3",
        "--nag-video-only",
    )
    assert kw["negative_prompt"] == "" and kw["nag"] == NAGConfig(scale=5.0, alpha=0.5, tau=3.0, audio=False)


@pytest.mark.parametrize("mode", ["--distilled", "--dfr"])
def test_cli_without_nag_passes_neither_kwarg(tmp_path, monkeypatch, mode):
    kw = _captured(tmp_path, monkeypatch, mode)
    assert "nag" not in kw and "negative_prompt" not in kw


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--distilled", "--negative-prompt", "n"], "--nag"),
        (["--dfr", "--negative-prompt", "n"], "--nag"),
        (["--distilled", "--nag"], "needs --negative-prompt"),
        (["--two-stage", "--nag", "--negative-prompt", "n"], "--distilled and --dfr"),
        (["--one-stage", "--nag", "--negative-prompt", "n"], "--distilled and --dfr"),
        (["--two-stages-hq", "--nag", "--negative-prompt", "n"], "--distilled and --dfr"),
        (["--distilled", "--nag-scale", "5"], "only apply with --nag"),
        (["--two-stage", "--nag-video-only"], "only apply with --nag"),
    ],
)
def test_cli_rejects_before_building_a_pipeline(tmp_path, monkeypatch, extra, match):
    with pytest.raises(SystemExit, match=match):
        _captured(tmp_path, monkeypatch, *extra)


def test_cli_rejects_a_bad_nag_value_before_building_a_pipeline(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="scale"):
        _captured(tmp_path, monkeypatch, "--distilled", "--nag", "--negative-prompt", "n", "--nag-scale", "0.5")
