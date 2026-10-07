"""Torch reference for the image-conditioning resize (``tests/test_image_preprocess.py``).

Standalone, NOT a pytest module: needs torch + upstream ltx-pipelines. Run it from an upstream
LTX-2 checkout's environment (v1.4.2 was used for the committed fixture)::

    cd /path/to/LTX-2 && uv run python /path/to/ltx-2-mlx/tests/parity_image_preprocess_reference.py \
      --out /path/to/ltx-2-mlx/tests/fixtures/image_preprocess_resize_golden.npz

Runs upstream ``ltx_pipelines.utils.media_io.resize.resize_and_center_crop`` (bilinear on float,
``align_corners=False``, no antialiasing) on seeded uint8 frames and stores inputs and outputs.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from ltx_pipelines.utils.media_io.resize import resize_and_center_crop

# (src_h, src_w, dst_h, dst_w): downscale, upscale, non-integer ratio with a crop on each axis.
CASES = ((37, 53, 24, 32), (19, 27, 40, 56), (61, 97, 40, 48), (50, 38, 32, 32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(188)
    arrays: dict[str, np.ndarray] = {}
    for i, (sh, sw, dh, dw) in enumerate(CASES):
        src = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)
        out = resize_and_center_crop(torch.tensor(src, dtype=torch.float32), dh, dw)  # (1, C, 1, H, W)
        arrays[f"src_{i}"] = src
        arrays[f"out_{i}"] = out[0, :, 0].permute(1, 2, 0).numpy().astype(np.float32)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
