# Model selection and optimization

## Decision

The deployment candidate is `garg_benchmark_seed42`, the five-channel cross-modal Swin-UNet student. It has the best archived three-seed all-water IoU and was also the fastest candidate in the focused CPU estimate. The initial two-channel U-Net hypothesis was therefore rejected by measurement.

This result does not imply operational flood-mapping readiness. The released Sen1Floods11 target is all water, and the selected model uses DEM, HAND, and a retrospective WorldCover water prior in addition to VV and VH.

## Candidate evidence

Accuracy is the archived 90-chip, three-seed global all-water IoU from the read-only research checkout at commit `9d9430fd1d53d7fbb21e63a4be2bd4b47177b7a6`. It was not recomputed on CPU.

Speed was measured with batch size 1 and 512 × 512 inputs. CPU values use one fixed image, one warm-up, two timed forwards, and 12 logical CPU threads in a fresh process per model. They are deliberately coarse estimates, not Cloud Run latency claims. GPU values use 10 warm-ups and 50 timed forwards on an RTX 3060 with explicit CUDA synchronization.

| Model/job | Architecture | Ch. | Three-seed IoU | Params | Checkpoint | CPU p50 | CPU p95 | GPU p50 | GPU p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unet_ch_01_seed42` | U-Net | 2 | 0.6804 | 31.0M | 372.6 MB | 32,753 ms | 42,442 ms | 46.9 ms | 50.0 ms |
| `unet_ch_01234_seed42` | U-Net | 5 | 0.6820 | 31.0M | 372.6 MB | 17,958 ms | 17,993 ms | 47.1 ms | 50.7 ms |
| `arch_resnet34_seed42` | ResNet34-U-Net | 5 | 0.6794 | 24.4M | 293.6 MB | 2,529 ms | 2,532 ms | 13.6 ms | 14.5 ms |
| `garg_benchmark_seed42` | Swin-UNet student | 5 | 0.7045 | 31.6M | 379.8 MB | 214 ms | 215 ms | 22.1 ms | 23.7 ms |

The selection rule retained models within 0.03 IoU of the best repeated mean. The two-channel U-Net was preferred only if its CPU estimate was within 1.5× of the fastest eligible model. It was not, so the fastest eligible model was selected. The CPU gaps are large, but two observations per model are only suitable for this coarse decision.

## ONNX export and fidelity

The selected checkpoint exported to a fixed-shape FP32 ONNX model with input `float32[1,5,512,512]` and output score map `float32[1,1,512,512]`. Across `Sri-Lanka_713926` and `India_80221`, ONNX versus PyTorch had maximum absolute score error `2.98e-6` and 100% agreement after thresholding at 0.5.

ONNX Runtime CPU estimates use one thread, two warm-ups per image, and five timed forwards per image. INT8 calibration and all fidelity checks use only those two images.

| Artifact | Size | CPU p50 | CPU p95 | Mask agreement vs FP32 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 ONNX | 134.95 MB | 767 ms | 772 ms | 100% | Deployment candidate |
| INT8 QDQ | 122.63 MB | 641 ms | 647 ms | 99.916% | Experimental |
| INT4 weight-only | 39.80 MB | 764 ms | 777 ms | 99.911% | Experimental |

INT8 was about 16% faster at p50 but reduced total size by only about 9%, while INT4 reduced size by about 71% without a material latency improvement. Neither quantized artifact is promoted: two images are insufficient for a defensible accuracy or calibration decision.

The private artifacts and hashes are recorded in `manifests/optimized_models.toml`; no checkpoint, ONNX file, calibration input, or score array is tracked by Git.

## Reproduction boundary

The serving application is reproducible with the committed `uv.lock`, `requirements.txt`, Dockerfile, manifests, and tests. The original training checkpoints and research dataset are intentionally not included in this repository.

If the private research artifacts are available, the tracked scripts provide the export and verification workflow:

```bash
uv run --locked python scripts/verify_optimized_artifacts.py
```

The model-selection values above should be interpreted with their stated protocols; they are not claims of operational flood-mapping performance or calibrated probability quality.
