# 🖼️ Batch Image Upscaler with Real-ESRGAN

A powerful Python script for **batch image upscaling** using **Real-ESRGAN** models with high-quality results.  
Supports **1×, 2×, and 3× scaling**, **in-place file replacement**, and automatic **RGB conversion** to ensure compatibility with platforms like Adobe Stock and Freepik.

---

## ✨ Features
- 🔄 **Batch Processing**: Process all images in a folder simultaneously
- 📈 **Flexible Scaling**: 1× (enhance without resize), 2×, or 3× (via 4× upscale + Lanczos downscale)
- 🖌️ **High Quality**: Uses pretrained Real-ESRGAN models (`general-x4v3`, `x4plus`, `x2plus`)
- 🔧 **In-Place Replacement**: Results directly overwrite original files (atomic replacement)
- 🧩 **Tiling Support**: Tile processing to save RAM/VRAM usage
- 🎨 **Auto RGB Conversion**: Converts colors to RGB for stock platform compatibility
- ⚡ **CPU/GPU Support**: Automatic GPU detection (CUDA) with CPU fallback option

---

## 📦 Requirements
- Python 3.8+
- [PyTorch](https://pytorch.org/get-started/locally/) (CPU/GPU compatible)
- Required libraries:
    ```
    realesrgan
    basicsr
    facexlib
    gfpgan
    opencv-python
    pillow
    tqdm
    ```

---

## 🔧 Installation

```bash
# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate   # Windows

# Install PyTorch (choose based on your GPU/CPU setup)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install realesrgan basicsr facexlib gfpgan opencv-python pillow tqdm
```

**Download Real-ESRGAN model weights** and place them in your project folder:
- `realesr-general-x4v3.pth`
- `RealESRGAN_x4plus.pth`
- `RealESRGAN_x2plus.pth`

---

## 🚀 Usage

### Basic Command
```bash
python batch_upscale_realesrgan.py /path/to/folder --scale 2 --model general-x4v3
```

### Command Line Options
```bash
--scale {1,2,3}                    # Output scale (1=enhance only, 2=2x, 3=3x)
--model {general-x4v3,x4plus,x2plus} # Choose model weights
--tile N                           # Tile size (e.g., 200-400, smaller = less RAM)
--fp16                             # Use half precision (saves VRAM, needs GPU support)
--cpu                              # Force CPU usage
--jpeg-quality N                   # JPEG quality override (default: 92)
--keep-metadata                    # Preserve metadata/EXIF data
```

### Examples
```bash
# Enhance without resizing
python batch_upscale_realesrgan.py ./images --scale 1 --model general-x4v3

# High-quality 2× upscale
python batch_upscale_realesrgan.py ./images --scale 2 --model general-x4v3

# 3× upscale with tile processing
python batch_upscale_realesrgan.py ./images --scale 3 --model general-x4v3 --tile 300
```

---

## ⚖️ Tips & Best Practices

- **⚠️ Backup**: Script overwrites original files. Modify code if you want separate output folder
- **🧩 Tiling**: Use `--tile 200-400` for high-resolution images to prevent out-of-memory errors
- **⚡ Performance**: Enable `--fp16` on supported GPUs for faster processing
- **🎯 Model Selection**:
    - `general-x4v3`: Versatile (photos + illustrations)
    - `x4plus`: Sharp results for photographs
    - `x2plus`: Native 2× upscaling

---

## 📄 License
This project uses Real-ESRGAN models. Please check the original [Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN) for licensing information.
