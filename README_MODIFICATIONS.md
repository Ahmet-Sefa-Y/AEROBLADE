# AEROBLADE - Custom Enhancements & Fork Documentation

> [!NOTE]
> This repository is a customized fork of the official [AEROBLADE (CVPR 2024)](https://github.com/jonasricker/aeroblade) project by Jonas Ricker, Denis Lukovnikov, and Asja Fischer.

---

## 🇹🇷 Türkçe Açıklama

Bu depo, CVPR 2024'te sunulan **AEROBLADE** projesinin özelleştirilmiş ve geliştirilmiş sürümüdür. Orijinal projeye atıfta bulunarak, platform bağımsız çalışma yetenekleri ve kullanım kolaylığı sağlayan ek geliştirmeler yapılmıştır.

### 🚀 Yapılan Özelleştirmeler ve Geliştirmeler

1. **Çoklu Platform ve Cihaz Desteği (CUDA / Apple Silicon MPS / CPU)**
   - `src/aeroblade/misc.py` içerisindeki `device()` fonksiyonu güncellenerek dinamik olarak CUDA, Apple Silicon (`mps`) veya CPU seçimi sağlanmıştır.
   - `torch.compile` ve `fp16` model yükleme adımları CUDA odaklı olmaktan çıkarılıp CPU ve MPS ortamlarında sorunsuz çalışacak şekilde düzenlenmiştir.

2. **Otomatik Görsel Boyutlandırma Kırpması (`crop_to_multiple_of_8`)**
   - Latent Diffusion AutoEncoder mimarilerinin gerektirdiği 8'in katı görsel boyutları için `src/aeroblade/data.py` içerisine `crop_to_multiple_of_8` merkezi kırpma fonksiyonu eklenmiştir.
   - Farklı çözünürlükteki görsellerin manuel müdahaleye gerek kalmadan işlenebilmesi için `DEFAULT_TRANSFORM` ve `high_level_funcs.py` veri akışlarına entegre edilmiştir.

3. **Çapraz Platform Bağımlılık Yönetimi (`requirements.txt`)**
   - Yalnızca Linux/CUDA sistemlerinde çalışan `triton` ve `nvidia-*` paketleri platform koşullarına (`platform_system == "Linux"`) bağlanmıştır. Böylece Windows ve macOS üzerinde `pip install -r requirements.txt` komutu hatasız çalışır.

4. **Depo Yapılandırması (`.gitignore` & `.gitattributes`)**
   - Çıktı klasörleri, önbellekler ve sanal ortamlar (`.venv`, `cache/`, `smoke_outputs/`, `aeroblade_output/`) depoya eklenmeyecek şekilde yapılandırılmıştır.
   - Satır sonu karakterleri ve ikili dosyalar için `.gitattributes` eklenmiştir.

---

## 🇬🇧 English Description

This repository is an enhanced fork of the **AEROBLADE** project. Below is a summary of custom modifications added to improve cross-platform compatibility and usability:

### Key Enhancements

- 🍏 **Cross-Platform Device Selection**: Automatically detects and uses **CUDA**, Apple Silicon (**MPS**), or **CPU** (`src/aeroblade/misc.py`).
- ✂️ **Automatic Image Preprocessing**: Center-crops arbitrary input images to dimensions divisible by 8 (`crop_to_multiple_of_8`) to satisfy AutoEncoder spatial requirements.
- 📦 **Platform-Conditional Dependencies**: Platform-specific dependency flags in `requirements.txt` prevent installation issues on Windows and macOS.
- 🧹 **Repository Cleanliness**: Configured `.gitignore` and `.gitattributes` for cross-platform Git workflow.

---

## Original Paper Citation

```bibtex
@inproceedings{ricker2024aeroblade,
  title={AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error},
  author={Ricker, Jonas and Lukovnikov, Denis and Fischer, Asja},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
