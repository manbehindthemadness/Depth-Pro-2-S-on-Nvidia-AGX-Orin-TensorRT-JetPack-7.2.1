# SPDX-License-Identifier: MIT
"""Compile a checked Depth Pro 2-S ONNX graph for the local Jetson GPU."""

import argparse
import ctypes
from pathlib import Path
import tempfile

import tensorrt as trt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    if args.onnx.suffix != ".onnx" or not args.onnx.is_file():
        parser.error("Input must be an existing .onnx file")
    if args.engine.suffix != ".engine":
        parser.error("Output must end in .engine")

    # This Linux process can otherwise produce a multi-GiB compiler core dump.
    if ctypes.CDLL(None).prctl(4, 0, 0, 0, 0) != 0:
        raise RuntimeError("Could not disable core dumps for the builder process")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = (str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        raise RuntimeError("\n".join(errors))

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError("Expected one RGB input and one inverse-depth output")
    rgb, depth = network.get_input(0), network.get_output(0)
    shape = tuple(rgb.shape)
    if len(shape) != 4 or shape[:2] != (1, 3) or min(shape) <= 0:
        raise ValueError(f"Expected static batch-one NCHW RGB, got {shape}")
    if rgb.name != "pixel_values" or rgb.dtype != trt.float32:
        raise ValueError("Expected FP32 pixel_values input")
    if depth.name != "inverse_depth" or depth.dtype != trt.float32 or tuple(depth.shape) != (1, *shape[2:]):
        raise ValueError("Expected FP32 inverse_depth matching the input height and width")

    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 2**20)
    print(f"Building {shape[3]}x{shape[2]} with TensorRT {trt.__version__}", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no engine")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.engine.parent, prefix=".build-") as directory:
        output = Path(directory) / args.engine.name
        output.write_bytes(memoryview(serialized))
        output.replace(args.engine)
    print(f"Saved {args.engine} ({args.engine.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
