# SPDX-License-Identifier: MIT
"""Check the exported ONNX and compare its CPU output against the PyTorch fixture."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import openvino as ov


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="ONNX export with a matching .npz fixture beside it")
    args = parser.parse_args()
    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)
    for initializer in model.graph.initializer:
        values = numpy_helper.to_array(initializer)
        if values.dtype.kind == "f" and (
            not np.isfinite(values).all() or np.max(np.abs(values), initial=0) > np.finfo(np.float16).max
        ):
            raise ValueError(f"Initializer outside finite FP16 range: {initializer.name}")
    del model

    core = ov.Core()
    graph = ov.convert_model(args.onnx)
    inference = core.compile_model(
        graph, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": 4}
    )
    with np.load(args.onnx.with_suffix(".npz")) as reference:
        for index, (pixels, expected) in enumerate(zip(reference["pixels"], reference["inverse_depth"])):
            depth = inference({"pixel_values": pixels[None]})[0]
            if depth.shape != (1, *expected.shape):
                raise ValueError(f"Expected shape {(1, *expected.shape)}, got {depth.shape}")
            np.testing.assert_allclose(depth[0], expected, atol=1e-4, rtol=2e-3)
            print(f"Sample {index} matches PyTorch, mean absolute error {np.abs(depth[0] - expected).mean():.8g}")


if __name__ == "__main__":
    main()
