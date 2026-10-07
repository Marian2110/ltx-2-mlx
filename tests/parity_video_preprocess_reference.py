"""Torch reference for the control-video preprocessing (``tests/test_video_preprocess.py``).

Standalone, NOT a pytest module: needs torch + PyAV + upstream ltx-pipelines. Run it from an
upstream LTX-2 checkout's environment (v1.4.2 was used for the committed fixture)::

    cd /path/to/LTX-2 && uv run python /path/to/ltx-2-mlx/tests/parity_video_preprocess_reference.py \\
      --video /path/to/ltx-2-mlx/tests/fixtures/video_preprocess_160x90_9f.mp4 \\
      --out /path/to/ltx-2-mlx/tests/fixtures/video_preprocess_golden.npz

Decodes the fixture clip with upstream ``decode_video_by_frame`` and runs ``video_preprocess``
(bilinear resize on floats + center crop + ``x / 127.5 - 1``) at targets whose aspect ratio differs
from the 16:9 source, storing the results as float16.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from ltx_pipelines.utils.media_io.decode import decode_video_by_frame, video_preprocess

# (height, width, starting_frame, frame_cap)
CASES = ((32, 32, 0, 5), (24, 48, 0, 5), (32, 16, 2, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    arrays: dict[str, np.ndarray] = {}
    device = torch.device("cpu")
    frames = list(decode_video_by_frame(args.video, device=device))
    decoded = torch.cat(frames).numpy()  # (F, H, W, 3) uint8
    arrays["decoded_count"] = np.array(len(frames))
    arrays["decoded_first_last"] = decoded[[0, -1]]
    for i, (h, w, start, cap) in enumerate(CASES):
        gen = decode_video_by_frame(args.video, device=device, starting_frame=start, frame_cap=cap)
        out = video_preprocess(gen, h, w, torch.float32, device)  # (1, 3, F, H, W)
        arrays[f"case_{i}"] = np.array([h, w, start, cap])
        # float16 keeps the fixture small; the MLX side returns bfloat16, so the test tolerance is wider anyway.
        arrays[f"out_{i}"] = out.numpy().astype(np.float16)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
