# AEROBLADE (Enhanced Cross-Platform Fork)

> **Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error**

This repository is an enhanced, cross-platform fork of the CVPR 2024 paper **AEROBLADE** by [Jonas Ricker](https://jonasricker.com), [Denis Lukovnikov](https://informatik.rub.de/ml/people/lukovnikov/), and [Asja Fischer](https://informatik.rub.de/fischer/).

<p align="center">
  <img src="media/header.png" width="70%" alt="AEROBLADE Header"> 
</p>

---

## 🌟 Key Improvements in this Fork

This version introduces several enhancements for seamless cross-platform execution and ease of use:

- 🍏 **Cross-Platform Device Auto-Selection:** Automatically detects and optimizes execution for **CUDA**, Apple Silicon (**MPS**), or **CPU** backends (`src/aeroblade/misc.py`).
- ✂️ **Automatic Image Preprocessing (`crop_to_multiple_of_8`):** Center-crops input images automatically to dimensions divisible by 8 so arbitrary image resolutions process without AutoEncoder spatial shape errors.
- 📦 **Cross-Platform Dependency Management:** Updated `requirements.txt` with platform environment markers (`platform_system == "Linux"`) allowing smooth installation on Windows, macOS, and Linux.
- 🧪 **Clean Project Structure:** Added `.gitignore` and `.gitattributes` configurations for clean Git repository tracking.

---

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ahmet-Sefa-Y/AEROBLADE.git
   cd AEROBLADE
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   # On Windows: .venv\Scripts\activate
   # On macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run AEROBLADE detection:**
   ```bash
   python scripts/run_aeroblade.py --files-or-dirs path/to/img.png path/to/dir
   ```
   - Running without arguments defaults to images in `example_images/`.
   - Images are automatically center-cropped to dimensions divisible by 8 (minimum 8x8 pixels).
   - Execution automatically utilizes **CUDA**, **Apple MPS**, or **CPU** in order of availability.
   - Computed distances are displayed and exported to `aeroblade_output/distances.csv`.

---

## 🔬 Reproducing Experiments

Here are the commands to reproduce the paper's experimental evaluations:

### Data Setup
Download the dataset from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10997234). Extract the archive and place the `data` directory inside the root of this repository.

#### Generated Images
Available in `data/raw/generated/`. Prompts are stored in `data/raw/prompts/`.

#### Real Images
Download LAION-5B real metadata and process:
```bash
img2dataset --url_list data/raw/real/real_metadata.parquet --input_format "parquet" --url_col "URL" --caption_col "TEXT" --output_folder tmp/laion --resize_mode "center_crop" --min_image_size 512 --max_image_area 589824 --image_size 512 --encode_format "png" --encode_quality 6
python scripts/rename_real_images.py
```

### Evaluation Scripts

- **Detection Performance:**
  ```bash
  python experiments/01_detect.py
  ```
  *(Add `--precomputed-real-dist data/precomputed/01_default_real_dist.pickle` to use pre-computed real image distances).*

- **Image Complexity Analysis:**
  ```bash
  python experiments/02_analyze_patches.py
  ```

- **Robustness to Perturbations:**
  ```bash
  python experiments/01_detect.py --experiment-id robustness --amount 250 --transforms clean jpeg_90 jpeg_80 jpeg_70 jpeg_60 jpeg_50 blur_1.0 blur_2.0 blur_3.0 blur_4.0 blur_5.0 crop_0.9 crop_0.8 crop_0.7 crop_0.6 crop_0.5 noise_0.05 noise_0.1 noise_0.15 noise_0.2 noise_0.25
  ```

- **Deeper Reconstructions:**
  ```bash
  python experiments/03_deeper_reconstructions.py --experiment-id deeper_sd15 --real-dir data/raw/real --fake-dir data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai --repo-id runwayml/stable-diffusion-v1-5
  ```

---

## 📜 Citation & Original Work

If you use AEROBLADE in your research, please cite the original paper:

```bibtex
@inproceedings{ricker2024aeroblade,
  title={AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error},
  author={Ricker, Jonas and Lukovnikov, Denis and Fischer, Asja},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
*Original Repository: [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade)*
