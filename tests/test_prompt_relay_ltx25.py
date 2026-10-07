"""Prompt Relay token ranges on LTX-2.5 packs (Gemma-4 tokenizer).

``map_token_ranges`` was written against the mlx-lm Gemma-3 tokenizer (2.3 packs). On 2.5 packs the
text encoder is :class:`Gemma4TextEncoder` with the pack's HuggingFace ``tokenizer.json``, which
behaves differently at the edges: it adds no BOS and no EOS, and ``<eos>`` is still id 1. These tests
pin the whole chain the relay mask relies on, with the real tokenizer:

1. each range covers exactly its local prompt in the combined prompt;
2. the ranges index the valid tokens that ``Gemma4TextEncoder.tokenize`` produces (left-padded to
   ``max_length``), so a change to that tokenization (e.g. a prepended BOS) fails here instead of
   silently shifting every segment by one column;
3. after the connector's front-packing (``_replace_padding_with_registers``) token *i* sits in column
   *i* of the ``Nk`` axis, which is where :func:`build_relay_mask` writes the penalty.

Skipped when the local 2.5 pack (``LTX25_Q8_DIR``) or ``transformers`` is absent. CPU only, no weights.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.conditioning.prompt_relay import build_relay_mask, map_token_ranges
from tests.conftest import LTX25_Q8_DIR

pytestmark = pytest.mark.skipif(
    LTX25_Q8_DIR is None or not (LTX25_Q8_DIR / "tokenizer.json").exists(),
    reason="local ltx-2.5-mlx-q8 pack (tokenizer.json) not found",
)

_MAX_LENGTH = 1024

# (global prompt, local prompts). Boundaries cover sentence punctuation, quoted speech, a local without
# final punctuation followed by a lowercase local, digits, non-ASCII text and repeated spaces.
_CASES = [
    (
        "Cinematic close-up of a woman in a sunlit kitchen, soft daylight.",
        ["She smiles warmly at the camera.", "She turns away and walks to the window."],
    ),
    (
        "A woman in a red sweater stands in a bright living room, static camera.",
        ['She looks at the camera and says: "Good morning!"', 'She waves and says: "See you tomorrow."'],
    ),
    (
        "wide shot of a harbour at dusk",
        ["two boats leave the pier", "a lighthouse turns on its beam", "3 seagulls land on the water"],
    ),
    (
        "Ein Café in Paris, Morgenlicht.",
        ["Une femme lit le journal  près de la fenêtre.", "日本の茶室で、男性がお茶を点てる。"],
    ),
]


@pytest.fixture(scope="module")
def encoder():
    transformers = pytest.importorskip("transformers")
    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4TextEncoder

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(LTX25_Q8_DIR))
    return Gemma4TextEncoder(tokenizer=tokenizer)


def _valid_ids(encoder, text: str) -> list[int]:
    """The valid (unpadded) token ids ``Gemma4TextEncoder.tokenize`` feeds the tower, in order."""
    token_ids, attention_mask = encoder.tokenize(text, max_length=_MAX_LENGTH)
    ids = np.array(token_ids)[0]
    mask = np.array(attention_mask)[0].astype(bool)
    # Left padding: every valid token sits after every pad.
    assert not mask[: int((~mask).sum())].any()
    return ids[mask].tolist()


@pytest.mark.parametrize(("global_prompt", "local_prompts"), _CASES)
def test_ranges_cover_exactly_each_local_prompt(encoder, global_prompt, local_prompts):
    tokenizer = encoder.tokenizer
    combined, ranges = map_token_ranges(tokenizer, global_prompt, local_prompts, max_length=_MAX_LENGTH)
    ids = _valid_ids(encoder, combined)

    assert len(ranges) == len(local_prompts)
    # Contiguous, ordered, and the last local ends on the last token (nothing appended after it).
    assert ranges[0][0] == len(_valid_ids(encoder, global_prompt))
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert end == start
    assert ranges[-1][1] == len(ids)
    for (start, end), local in zip(ranges, local_prompts):
        assert tokenizer.decode(ids[start:end]) == " " + local


def test_gemma4_tokenizer_edge_assumptions(encoder):
    """The facts ``map_token_ranges`` relies on for this tokenizer: no BOS, no EOS added by ``encode``."""
    tokenizer = encoder.tokenizer
    probe = tokenizer.encode("test")
    assert probe and probe[0] != tokenizer.bos_token_id
    assert probe[-1] != tokenizer.eos_token_id
    # ``encode`` is what the relay measures, ``tokenize`` is what the encoder runs: same ids.
    text = "A woman smiles. She waves."
    assert _valid_ids(encoder, text) == tokenizer.encode(text)


@pytest.mark.parametrize(("global_prompt", "local_prompts"), _CASES[:2])
def test_ranges_land_on_mask_columns_after_front_packing(encoder, global_prompt, local_prompts):
    """Feed token ids through the connector's packing and check the mask penalises those columns."""
    from ltx_core_mlx.text_encoders.gemma.embeddings_connector import _replace_padding_with_registers

    combined, ranges = map_token_ranges(encoder.tokenizer, global_prompt, local_prompts, max_length=_MAX_LENGTH)
    token_ids, attention_mask = encoder.tokenize(combined, max_length=_MAX_LENGTH)

    # One feature per position carrying the token id, so the packed layout is directly readable.
    hidden = token_ids.astype(mx.float32)[:, :, None]
    registers = mx.full((1, 128, 1), -1.0)
    packed = np.array(_replace_padding_with_registers(hidden, attention_mask, registers))[0, :, 0]

    ids = _valid_ids(encoder, combined)
    np.testing.assert_array_equal(packed[: len(ids)], np.array(ids, dtype=np.float32))
    assert np.all(packed[len(ids) :] == -1.0)  # registers fill the tail

    # Two segments of 8 latent frames, 1 token per frame. Frame 4 is segment 0's midpoint (free for its
    # tokens) and 8 frames from segment 1's: the penalised columns must be exactly segment 1's tokens,
    # i.e. its local prompt once read back from the packed sequence.
    mask = np.array(
        build_relay_mask(
            token_ranges=ranges,
            segment_lengths=[8, 8],
            num_video_tokens=16,
            tokens_per_frame=1,
            latent_frames=16,
            num_text_tokens=_MAX_LENGTH,
            dtype=mx.float32,
        )
    )[0, 0]
    penalised = np.nonzero(mask[4] < 0)[0]
    assert penalised.tolist() == list(range(*ranges[1]))
    assert encoder.tokenizer.decode(packed[penalised].astype(int).tolist()) == " " + local_prompts[1]
