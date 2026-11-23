# Depth Pro 2-S TensorRT engines for Jetson AGX Orin

An independent conversion of Apple Depth Pro 2-S for single-frame inference on the Jetson AGX Orin GPU. The first published build and engine set is tagged `v1.0.0`.

**Research use.** The engines are model derivatives of Apple Depth Pro 2-S. Apple's agreement limits their use and redistribution to noncommercial scientific research and academic development as defined in the agreement. [Apple model license](licenses/APPLE-MODEL.txt)

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

## Versions and compatibility

| Component | Tested version |
| --- | --- |
| Target | NVIDIA Jetson AGX Orin (Linux aarch64, SM 8.7) |
| JetPack | 7.2.1, package `7.2.1-b49` |
| Jetson Linux | 39.2.1 |
| CUDA | 13.2 |
| TensorRT | 10.16.2.10, package `10.16.2.10-1+cuda13.2` |
| Depth Pro 2 source | `aa389cbb0b74341664ed663a0966c859905cd6c0` |
| Depth Pro 2-S checkpoint | `903a1c1ade918831d138adcc4aedbcc8f4da0078` |
| CPU exporter | Python 3.12, PyTorch `2.9.1+cpu`, ONNX opset 17 |

Use the serialized engines on a matching GPU and runtime. Rebuild on the deployment target when the GPU architecture or TensorRT version differs. [NVIDIA engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html#software-and-system-requirements)

## Prebuilt engines

Each archive contains one TensorRT `.engine`, the Apple model agreement, a notice describing the conversion, upstream acknowledgements, and citation metadata.

| Resolution (width × height) | Download |
| --- | --- |
| 320×200 | [depthpro2-s-320x200-fp16.tar.gz](engines/depthpro2-s-320x200-fp16.tar.gz) |
| 640×400 | [depthpro2-s-640x400-fp16.tar.gz](engines/depthpro2-s-640x400-fp16.tar.gz) |
| 864×544 | [depthpro2-s-864x544-fp16.tar.gz](engines/depthpro2-s-864x544-fp16.tar.gz) |
| 1280×800 | [depthpro2-s-1280x800-fp16.tar.gz](engines/depthpro2-s-1280x800-fp16.tar.gz) |

Use the [archive checksums](engines/SHA256SUMS) or the [extracted engine checksums](engines/ENGINE-SHA256SUMS) to verify downloads.

```sh
(cd engines && sha256sum --check SHA256SUMS)
mkdir -p work/prebuilt-864x544
tar -xzf engines/depthpro2-s-864x544-fp16.tar.gz -C work/prebuilt-864x544
trtexec --loadEngine=work/prebuilt-864x544/depthpro2-s-864x544-fp16.engine --warmUp=500 --duration=3
```

| Tensor | Type | Shape |
| --- | --- | --- |
| `pixel_values` | FP32 planar RGB in `[0, 1]` | `[1, 3, H, W]` |
| `inverse_depth` | FP32 focal-normalized inverse depth | `[1, H, W]` |

The graph applies its own channel normalization and uses mixed precision internally. It processes a single frame with no temporal history. The export omits the separate focal-length estimator. For positive inverse depth, use `depth_m = fx_px / (W * inverse_depth)`, where `fx_px` is the horizontal focal length in pixels at the cropped and resized input. [Apple model documentation](https://github.com/apple-aiml-research/ml-depth-pro-2/tree/aa389cbb0b74341664ed663a0966c859905cd6c0)

## TensorRT versus PyTorch CUDA

Both backends ran on the same Jetson AGX Orin GPU with the same batch-one RGB inputs. The CUDA baseline used eager PyTorch `2.12.1+cu132` and cuDNN `9.20.0`, with Apple's original model weights, single-frame inference, and the focal-length estimator disabled. Each latency is the median of 72 runs after warmup, including transfers through preallocated pinned host buffers.

| Resolution | TensorRT | PyTorch CUDA BF16 | PyTorch CUDA FP16 | TensorRT speedup vs BF16 |
| --- | ---: | ---: | ---: | ---: |
| 320×200 | 4.12 ms | 34.16 ms | 34.12 ms | 8.29× |
| 640×400 | 10.80 ms | 36.93 ms | 37.13 ms | 3.42× |
| 1280×800 | 49.81 ms | 77.21 ms | 75.84 ms | 1.55× |

With inputs and outputs resident on the GPU, the CUDA-event medians were as follows.

| Resolution | TensorRT | PyTorch CUDA BF16 | PyTorch CUDA FP16 |
| --- | ---: | ---: | ---: |
| 320×200 | 4.02 ms | 34.06 ms | 33.94 ms |
| 640×400 | 10.53 ms | 36.87 ms | 37.11 ms |
| 1280×800 | 48.99 ms | 76.50 ms | 75.01 ms |

The ARM64 PyTorch wheel reported a support warning for Orin's SM 8.7. Both eager modes executed on CUDA and returned finite depth maps on the validation inputs. The timings describe that build of PyTorch.

The separate 864×544 TensorRT test measured 20.22 ms median including transfers.

## Build on Jetson

Use Python 3.12, the JetPack TensorRT runtime and Python bindings, and the pinned Apple source and checkpoint. The CPU export and verification run in separate virtual environments. The final compile uses the system Python with JetPack's TensorRT bindings.

```sh
mkdir -p third_party work
git clone https://github.com/apple-aiml-research/ml-depth-pro-2.git third_party/ml-depth-pro-2
git -C third_party/ml-depth-pro-2 checkout --detach aa389cbb0b74341664ed663a0966c859905cd6c0
curl --fail --location \
  https://huggingface.co/apple/DepthPro2-S/resolve/903a1c1ade918831d138adcc4aedbcc8f4da0078/model.safetensors \
  --output work/depthpro2-s.safetensors

python3.12 -m venv .venv-export
.venv-export/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
.venv-export/bin/python -m pip install ./third_party/ml-depth-pro-2 \
  numpy==2.5.4 timm==1.0.31 onnx==1.23.2 safetensors==0.8.0

python3.12 -m venv .venv-check
.venv-check/bin/python -m pip install numpy==1.26.4 onnx==1.23.2 openvino==2026.4.1
```

Build one fixed-size engine. Change `--size` and the filename stem to build another resolution in the table.

```sh
stem=work/depthpro2-s-864x544
.venv-export/bin/python scripts/export_depth.py \
  --apple-source third_party/ml-depth-pro-2 \
  --checkpoint work/depthpro2-s.safetensors --size 864 544 --output "$stem"
.venv-check/bin/python scripts/check_onnx.py "$stem.onnx"
/usr/bin/python3 -u scripts/build_engine.py "$stem.onnx" "$stem.engine"
trtexec --loadEngine="$stem.engine" --warmUp=500 --duration=3
```

The exporter verifies the checkpoint SHA-256, uses Apple's example image and seeded noise to fit FP16 ranges, and checks the fitted graph against PyTorch. The ONNX check evaluates both inputs in OpenVINO CPU at FP32 before compilation. The builder enables FP16 tactics, uses a 1 GiB workspace limit, and selects optimization level 0. Builder crashes were observed during testing, and level 0 produced the distributed engines. TensorRT can choose different tactics when rebuilding the same graph. In one verification run, a 320×200 rebuild from byte-identical ONNX passed the numerical checks but ran at 9.49 ms median. The bundled 320×200 engine measured 4.41 ms in the same validation harness. [NVIDIA tactic selection](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)

## Attribution and licenses

The original scripts under `scripts/` are under the [MIT license for the build scripts](licenses/BUILD-SCRIPTS-MIT.txt). Apple's upstream software has its own [software terms](licenses/APPLE-SOFTWARE.txt). The [Apple model license](licenses/APPLE-MODEL.txt) applies to the model derivatives. Include it and the [derivative notice](engines/NOTICE.txt) when redistributing an engine.

Credit Peiyun Hu, Marko Mihajlovic, Lyndon Duong, Lahav Lipson, Wei Dong, Amaël Delaunoy, Stephan R. Richter, and Vladlen Koltun for Depth Pro 2. Apple's acknowledgements credit DINOv3 and timm. [Paper](https://arxiv.org/abs/2611.26603) · [Apple model card](https://huggingface.co/apple/DepthPro2) · [Apple acknowledgements](licenses/APPLE-ACKNOWLEDGEMENTS.txt) · [Citation metadata](licenses/APPLE-CITATION.cff)
