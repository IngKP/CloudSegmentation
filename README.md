# CloudSegmentation

A compact, production-style ML project that turns an existing Sentinel-1 SAR
research checkpoint into a reproducible CPU inference service and interactive
flood-event demo.

The Sen1Floods11 reference target used by this project is an **all-water mask**,
not a transient-inundation-only target. The project demonstrates model
selection, inference optimization, containerization, API serving, cloud
operations, and honest model limitations; it does not claim operational flood
mapping readiness.

## Current phase

Environment bootstrap and model audit. The initial benchmark compares a focused
accuracy/runtime Pareto set:

- 2-channel U-Net (VV + VH)
- 5-channel U-Net (VV + VH + DEM + HAND + Water)
- 5-channel ResNet34-U-Net
- 5-channel cross-modal Swin-UNet student

The 2-channel U-Net is the leading deployment hypothesis because its repeated
accuracy is within roughly 0.024 IoU of the best cross-modal mean while avoiding
all ancillary inputs. CPU measurements and ONNX fidelity checks must confirm the
choice.

## Setup

See [`docs/setup-pc-pop.md`](docs/setup-pc-pop.md). Large research artifacts are
private, untracked, and verified from [`manifests/models.toml`](manifests/models.toml).

## Project direction

The complete implementation plan is in [`plan.md`](plan.md). The next milestone
is reproducible PyTorch inference and a CPU benchmark for the four staged model
representatives.

## License

Project source code is available under the MIT License. Dataset and model
artifacts retain their own terms and are not covered by that license.

