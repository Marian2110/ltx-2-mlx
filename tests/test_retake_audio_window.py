"""retake's audio window: the audio tokens whose time interval overlaps the latent-frame window (upstream
``TemporalRegionMask``), not a proportional frames-to-tokens split."""

from __future__ import annotations

import pytest

from ltx_core_mlx.utils.positions import compute_audio_token_count
from ltx_pipelines_mlx.retake import audio_window_tokens


def _overlap_reference(start_frame, end_frame, num_tokens, fps):
    """Upstream's rule, written out: pixel-frame bounds of the latent frames vs each token's mel-frame bounds."""
    t0 = max(0, 8 * start_frame - 7) / fps
    t1 = max(0, 8 * end_frame - 7) / fps
    return [i for i in range(num_tokens) if max(0, 4 * i + 1) * 0.01 > t0 and max(0, 4 * i - 3) * 0.01 < t1]


def test_49_frame_example_starts_at_token_10():
    num_tokens = compute_audio_token_count(49, frame_rate=24.0)
    assert num_tokens == 51
    assert audio_window_tokens(2, 5, num_tokens, 24.0) == (10, 36)  # proportional mapping gave (15, 36)


def test_whole_clip_and_last_frame():
    num_tokens = compute_audio_token_count(49, frame_rate=24.0)
    assert audio_window_tokens(0, 7, num_tokens, 24.0) == (0, num_tokens)
    start, end = audio_window_tokens(6, 7, num_tokens, 24.0)
    assert end == num_tokens and start > 0


@pytest.mark.parametrize("frames, fps", [(49, 24.0), (97, 24.0), (121, 25.0), (241, 30.0), (25, 24.0)])
def test_matches_the_overlap_rule_on_every_window(frames, fps):
    latent_frames = (frames - 1) // 8 + 1
    num_tokens = compute_audio_token_count(frames, frame_rate=fps)
    for s in range(latent_frames):
        for e in range(s + 1, latent_frames + 1):
            inside = _overlap_reference(s, e, num_tokens, fps)
            assert audio_window_tokens(s, e, num_tokens, fps) == (inside[0], inside[-1] + 1)
            assert inside == list(range(inside[0], inside[-1] + 1))  # contiguous
