import os
import sys
import argparse
import tempfile
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from realesrgan import RealESRGAN

# --- Util ---
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in VALID_EXT and p.is_file():
            yield p

def ensure_rgb(pil_img: Image.Image) -> Image.Image:
    # Normalisasi warna agar tidak "terdeteksi CMYK" di platform stok
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")
    # Hilangkan profil ICC yang bikin ribet kompatibilitas
    pil_img.info.pop("icc_profile", None)
    return pil_img

def atomic_replace(src_tmp: Path, dst: Path):
    # Os.replace = atomic rename di mayoritas OS
    os.replace(src_tmp, dst)

# --- SR core ---
def load_model(device, model_name: str, scale_native: int):
    """
    model_name: 'general-x4v3' | 'x4plus' | 'x2plus'
    scale_native: 4 untuk x4, 2 untuk x2
    """
    model = RealESRGAN(device, scale=scale_native)
    weight_map = {
        "general-x4v3": "realesr-general-x4v3.pth",
        "x4plus": "RealESRGAN_x4plus.pth",
        "x2plus": "RealESRGAN_x2plus.pth",
    }
    weights = weight_map[model_name]
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"Bobot '{weights}' tidak ditemukan. Letakkan file bobot di folder skrip."
        )
    model.load_weights(weights)
    return model

def enhance_with_scale(model, img: Image.Image, out_scale: int, native_scale: int):
    """
    Jalankan enhance dengan model native (2× atau 4×), lalu resize ke target 1/2/3×.
    out_scale: 1 | 2 | 3
    native_scale: 2 | 4
    """
    # Safety: untuk 1× kita tetap enhance agar noise/tekstur diperbaiki, lalu kembalikan ke ukuran asli
    # Strategi: enhance sekali ke native_scale, lalu down/up ke target out_scale menggunakan Lanczos.
    w, h = img.width, img.height
    # Upscale ke native
    img_sr = model.predict(img)  # hasil w*native_scale, h*native_scale

    # Hitung target size
    target_w = int(round(w * out_scale))
    target_h = int(round(h * out_scale))

    if img_sr.width == target_w and img_sr.height == target_h:
        return img_sr

    # Resize ke target menggunakan Lanczos (tajam & aman untuk downsample)
    img_out = img_sr.resize((target_w, target_h), Image.LANCZOS)
    return img_out

def process_one(
    path: Path,
    model,
    target_scale: int,
    native_scale: int,
    jpeg_quality: int = 92,
    keep_metadata: bool = False,
):
    img = Image.open(path)
    img = ensure_rgb(img)

    out_img = enhance_with_scale(model, img, target_scale, native_scale)

    # Simpan atomically
    suffix = path.suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
    save_kwargs = {}
    if suffix in (".jpg", ".jpeg"):
        save_kwargs["quality"] = jpeg_quality
        save_kwargs["optimize"] = True
        save_kwargs["progressive"] = True
    if not keep_metadata:
        out_img.info.clear()
    out_img.save(tmp_path, **save_kwargs)
    atomic_replace(tmp_path, path)

def main():
    ap = argparse.ArgumentParser(
        description="Batch upscale Real-ESRGAN (1x/2x/3x) dengan replace in-place."
    )
    ap.add_argument("input_dir", type=str, help="Folder berisi gambar")
    ap.add_argument(
        "--scale",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Skala output: 1 (enhance saja), 2, atau 3",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="general-x4v3",
        choices=["general-x4v3", "x4plus", "x2plus"],
        help="Pilih bobot model",
    )
    ap.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Ukuran tile (0=auto by library; kecilkan jika RAM/GPU mepet, mis. 200~400)",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="Gunakan half precision (hemat VRAM; butuh GPU yang mendukung)",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="Kualitas JPEG saat overwrite (jpg/jpeg saja)",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Paksa pakai CPU meski ada GPU",
    )
    ap.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Pertahankan metadata/EXIF (default: dibuang untuk kompatibilitas platform stok)",
    )

    args = ap.parse_args()
    root = Path(args.input_dir)
    if not root.exists():
        print(f"Folder '{root}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    # Tentukan device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tentukan native scale dari model
    native_scale = 4 if args.model in ("general-x4v3", "x4plus") else 2

    # Load model
    model = load_model(device, args.model, native_scale)

    # Atur tiling & precision jika tersedia (opsi wrapper)
    # Catatan: wrapper RealESRGAN ini otomatis tiling; sebagian parameter internal tidak exposed.
    # Gunakan --tile untuk menekan penggunaan memori via chunking internal jika tersedia.
    if hasattr(model, "tile"):
        model.tile = args.tile if args.tile and args.tile > 0 else 0
    if args.fp16 and device.type == "cuda" and hasattr(model, "model"):
        model.model.half()

    files = list(list_images(root))
    if not files:
        print("Tidak ada gambar yang ditemukan.")
        return

    print(f"Device      : {device}")
    print(f"Model       : {args.model} (native x{native_scale})")
    print(f"Target scale: {args.scale}x")
    print(f"Tile        : {args.tile}")
    print(f"FP16        : {args.fp16}")
    print(f"Files       : {len(files)}")

    for p in tqdm(files, desc="Upscaling", unit="img"):
        try:
            process_one(
                p,
                model=model,
                target_scale=args.scale,
                native_scale=native_scale,
                jpeg_quality=args.jpeg_quality,
                keep_metadata=args.keep_metadata,
            )
        except Exception as e:
            print(f"[SKIP] {p.name} error: {e}")

    print("Selesai ✅")

if __name__ == "__main__":
    main()
