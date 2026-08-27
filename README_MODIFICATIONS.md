# AEROBLADE - Custom Enhancements & Fork Documentation

> [!NOTE]
> This repository is an enhanced, cross-platform fork of the official [AEROBLADE (CVPR 2024)](https://github.com/jonasricker/aeroblade) project by Jonas Ricker, Denis Lukovnikov, and Asja Fischer.

---

## 🚀 Key Improvements and Modifications

1. **Cross-Platform Device Auto-Selection (`CUDA` / Apple Silicon `MPS` / `CPU`)**
   - Updated `device()` in `src/aeroblade/misc.py` to dynamically detect and prioritize **CUDA**, Apple Silicon (**MPS**), or **CPU**.
   - Refactored `torch.compile` calls and precision settings in `src/aeroblade/image.py` and `src/aeroblade/distances.py` so the codebase executes smoothly across macOS and CPU-only environments without CUDA dependencies.

2. **Automatic Image Dimension Preprocessing (`crop_to_multiple_of_8`)**
   - Implemented `crop_to_multiple_of_8` center-cropping in `src/aeroblade/data.py` to ensure input images conform to the 8-pixel grid requirements of Latent Diffusion AutoEncoders.
   - Integrated into `DEFAULT_TRANSFORM` and processing flows (`high_level_funcs.py`), enabling arbitrary image inputs without manual image dimension adjustments.

3. **Cross-Platform Dependency Management (`requirements.txt`)**
   - Configured Linux-only dependencies (`triton`, `nvidia-*`) with environment markers (`platform_system == "Linux"`) so `pip install -r requirements.txt` succeeds cleanly on Windows, macOS, and Linux.

4. **Git Repository Hygiene (`.gitignore` & `.gitattributes`)**
   - Added `.gitattributes` for standardized cross-platform line endings and binary media rules.
   - Expanded `.gitignore` to prevent generated artifacts, pre-computed outputs (`smoke_outputs`, `full_outputs`, `aeroblade_output`, `cache`), and virtual environments (`.venv`) from bloating the Git index.

---

## Citation & Original Work

```bibtex
@inproceedings{ricker2024aeroblade,
  title={AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error},
  author={Ricker, Jonas and Lukovnikov, Denis and Fischer, Asja},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
*Original Repository: [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade)*
