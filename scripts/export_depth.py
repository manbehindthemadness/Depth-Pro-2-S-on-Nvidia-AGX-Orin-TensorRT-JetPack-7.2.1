# SPDX-License-Identifier: MIT
"""Export a static, single-frame Depth Pro 2-S model and a PyTorch fixture."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import load_file
import torch
from torch import nn

CHECKPOINT_SHA256 = "583c4d630779fd231f969c738733624963ad34bd3c29f9c64088257e3436f97c"

class ExportModel(nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, pixels):
        return self.step(pixels)

def frames(height, width, example):
    with Image.open(example) as file:
        rgb = ImageOps.fit(file.convert("RGB"), (width, height))
        image = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(height, width, 3)
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255
    noise = np.random.default_rng(42).random((3, height, width), dtype=np.float32)
    return torch.from_numpy(np.stack((image, noise)))

def main():
    parser = argparse.ArgumentParser(description="Export a static single-frame Depth Pro 2-S graph.")
    parser.add_argument("--apple-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem without a file extension")
    args = parser.parse_args()
    width, height = args.size
    if min(width, height) < 32:
        parser.error("width and height must each be at least 32")
    with args.checkpoint.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint does not match the pinned Apple Depth Pro 2-S weights")

    apple_source = args.apple_source.resolve()
    sys.path[:0] = [str(apple_source / "src"), str(apple_source)]
    from depth_pro_2 import DepthPro2
    from depth_pro_2.model import PRESETS
    from demo.export.fp16 import fit_attention_fp16, fit_fp16
    from demo.export.stream import CausalStep

    torch.set_num_threads(4)
    preset = PRESETS["depthpro2-s"]
    model = DepthPro2(
        backbone_preset=preset.backbone_preset,
        reassembled_blocks=preset.reassembled_blocks,
        cross_frame_blocks=preset.cross_frame_blocks,
        dims_reassemble=preset.dims_reassemble,
        dim_decoder=preset.dim_decoder,
        fov=False,
        precision=torch.float32,
    ).eval()
    model.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    pixels = frames(height, width, apple_source / "assets/example.jpg")
    with torch.inference_mode():
        ref = torch.cat([CausalStep(model, 1, height, width).eval()(image[None]) for image in pixels])
        residual_divisor = fit_fp16(model, pixels)
        divisors = fit_attention_fp16(model, pixels, 1)
        network = ExportModel(CausalStep(model, 1, height, width, divisors)).eval()
        result = torch.cat([network(image[None]) for image in pixels])
        torch.testing.assert_close(result, ref, atol=1e-4, rtol=1e-3)
        name = args.output
        name.parent.mkdir(parents=True, exist_ok=True)
        np.savez(name.with_suffix(".npz"), pixels=pixels.numpy(), inverse_depth=ref.numpy())
        torch.onnx.export(
            network, (pixels[:1],), name.with_suffix(".onnx"),
            dynamo=False, opset_version=17,
            input_names=["pixel_values"], output_names=["inverse_depth"],
        )
    print(f"Exported {width}x{height}, residual divisor {residual_divisor}, logit divisors {divisors}")

if __name__ == "__main__":
    main()
