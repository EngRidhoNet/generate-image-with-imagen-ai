import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageFilter, ImageCms
from tqdm import tqdm

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def list_images(root: Path):
    """Find all image files in directory"""
    for p in root.rglob("*"):
        if p.suffix.lower() in VALID_EXT and p.is_file():
            yield p

# -------------------- ICC Helpers --------------------

KNOWN_ICC_LOCATIONS = {
    "mac": [
        "/System/Library/ColorSync/Profiles",
        "/Library/ColorSync/Profiles",
        str(Path.home() / "Library/ColorSync/Profiles"),
    ],
    "win": [
        r"C:\Windows\System32\spool\drivers\color",
    ],
    "linux": [
        "/usr/share/color/icc",
        "/usr/local/share/color/icc",
        str(Path.home() / ".color/icc"),
    ],
}

ICC_FILENAMES = {
    "adobe": [
        "AdobeRGB1998.icc",
        "Adobe RGB (1998).icc",
        "AdobeRGB1998.icm",
        "Adobe RGB (1998).icm",
    ],
    "prophoto": [
        "ProPhoto.icc",
        "ProPhoto RGB.icc",
        "ProPhoto.icm",
        "ProPhoto RGB.icm",
        "ROMM RGB.icc",
    ],
}

def detect_platform():
    if sys.platform.startswith("darwin"):
        return "mac"
    if sys.platform.startswith("win"):
        return "win"
    return "linux"

def find_icc_file(profile_key: str, icc_dir: Path | None) -> Path | None:
    """Try to find ICC file for AdobeRGB/ProPhoto."""
    candidates = []
    if icc_dir:
        for name in ICC_FILENAMES.get(profile_key, []):
            candidates.append(Path(icc_dir) / name)

    plat = detect_platform()
    for base in KNOWN_ICC_LOCATIONS.get(plat, []):
        for name in ICC_FILENAMES.get(profile_key, []):
            candidates.append(Path(base) / name)

    for c in candidates:
        if c.exists():
            return c
    return None

def get_profile_and_bytes(target_profile: str, icc_dir: Path | None):
    """
    Returns (target_profile_obj, icc_bytes) for embedding.
    For sRGB, uses built-in profile. For others, tries to load ICC file.
    """
    target_profile = target_profile.lower()
    if target_profile == "srgb":
        prof = ImageCms.createProfile("sRGB")
        # Build a temp image to extract profile bytes cleanly
        # (Pillow doesn't expose direct bytes from createProfile)
        dummy = Image.new("RGB", (1, 1))
        converted = dummy.copy()
        icc_bytes = None
        # Fallback: synthesize an sRGB profile via buildTransform roundtrip
        try:
            # Create a transform to sRGB (no-op but preserves icc embedding)
            tr = ImageCms.buildTransformFromOpenProfiles(prof, prof, "RGB", "RGB")
            converted = ImageCms.applyTransform(dummy, tr)
            icc_bytes = converted.info.get("icc_profile", None)
        except Exception:
            icc_bytes = None
        return prof, icc_bytes

    key = "adobe" if target_profile == "adobe" else "prophoto"
    path = find_icc_file(key, icc_dir)
    if not path:
        return None, None

    try:
        prof = ImageCms.getOpenProfile(str(path))
        with open(path, "rb") as f:
            icc_bytes = f.read()
        return prof, icc_bytes
    except Exception:
        return None, None

def build_transform_for_output(src_img: Image.Image, dst_profile) -> Image.Image:
    """
    Build transform from source profile (embedded or assume sRGB) to dst_profile.
    Returns an RGB image in destination color space.
    """
    # Ensure RGB for conversion pipeline
    img = src_img
    if img.mode not in ["RGB", "RGBA"]:
        img = img.convert("RGB")

    src_profile = None
    try:
        if "icc_profile" in img.info and img.info["icc_profile"]:
            src_profile = ImageCms.ImageCmsProfile(io=img.info["icc_profile"])
        else:
            src_profile = ImageCms.createProfile("sRGB")
    except Exception:
        src_profile = ImageCms.createProfile("sRGB")

    # If has alpha, flatten to white (JPEG has no alpha)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg

    # Convert RGB->RGB using profiles
    try:
        transform = ImageCms.buildTransformFromOpenProfiles(
            src_profile, dst_profile, "RGB", "RGB", intent=0, flags=ImageCms.FLAGS_SOFTPROOFING & 0
        )
        out = ImageCms.applyTransform(img, transform)
        return out
    except Exception:
        # If transform fails, fallback to simple RGB (will still embed target ICC if available)
        return img.convert("RGB")

# -------------------- Image ops --------------------

def upscale_image_pil(input_path: Path, output_path: Path, scale: int, method="lanczos"):
    """Upscale image using PIL with high-quality resampling"""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB/RGBA compatibility
            if img.mode not in ['RGB', 'RGBA']:
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')

            width, height = img.size
            new_width = width * scale
            new_height = height * scale

            if method == "lanczos":
                resample = Image.LANCZOS
            elif method == "bicubic":
                resample = Image.BICUBIC
            elif method == "nearest":
                resample = Image.NEAREST
            else:
                resample = Image.LANCZOS

            upscaled = img.resize((new_width, new_height), resample)

            # Mild sharpening for lanczos
            if method == "lanczos":
                upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=0))

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Preserve PNG alpha in non-JPEG; for JPEG flatten to white
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                if upscaled.mode == "RGBA":
                    bg = Image.new("RGB", upscaled.size, (255, 255, 255))
                    bg.paste(upscaled, mask=upscaled.split()[-1])
                    upscaled = bg
                upscaled.save(output_path, 'JPEG', quality=95, optimize=True)
            else:
                upscaled.save(output_path, optimize=True)

            return True, "Success"

    except Exception as e:
        return False, str(e)

def upscale_image_sips(input_path: Path, output_path: Path, scale: int):
    """Use macOS built-in sips command for upscaling (high quality)"""
    try:
        import subprocess
        with Image.open(input_path) as img:
            width, height = img.size

        new_width = width * scale
        new_height = height * scale

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "sips",
            "-z", str(new_height), str(new_width),
            str(input_path),
            "--out", str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr

    except Exception as e:
        return False, str(e)

def convert_to_jpg_with_profile(input_path: Path, output_path: Path, profile="srgb", icc_dir: Path | None = None, quality: int = 95, assume_input_srgb: bool = True):
    """
    Convert image to JPEG with specified color profile (srgb, adobe, prophoto).
    - Will embed ICC if found/available.
    - If image has alpha, it will be flattened to white.
    """
    try:
        # Prepare target profile + bytes for embedding
        dst_prof, dst_icc_bytes = get_profile_and_bytes(profile, icc_dir)
        if profile.lower() != "srgb" and (dst_prof is None or dst_icc_bytes is None):
            return False, f"ICC for profile '{profile}' not found. Provide --icc-dir pointing to the ICC files."

        with Image.open(input_path) as img:
            # If no embedded ICC and assume_input_srgb, spoof sRGB source
            if assume_input_srgb and "icc_profile" not in img.info:
                # Attach a synthetic sRGB source by converting via a no-op
                # (handled inside build_transform_for_output)
                pass

            # Transform to destination color space
            if profile.lower() == "srgb":
                dst_prof = ImageCms.createProfile("sRGB")
                # Try to synthesize icc bytes for embedding
                if not dst_icc_bytes:
                    try:
                        dummy = Image.new("RGB", (1,1))
                        tr = ImageCms.buildTransformFromOpenProfiles(dst_prof, dst_prof, "RGB", "RGB")
                        dst_icc_bytes = ImageCms.applyTransform(dummy, tr).info.get("icc_profile", None)
                    except Exception:
                        dst_icc_bytes = None

            converted = build_transform_for_output(img, dst_prof)

            # Ensure RGB (JPEG)
            if converted.mode != "RGB":
                converted = converted.convert("RGB")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = dict(quality=quality, optimize=True)
            if dst_icc_bytes:
                save_kwargs["icc_profile"] = dst_icc_bytes

            converted.save(output_path, "JPEG", **save_kwargs)
            return True, "Success"

    except Exception as e:
        return False, str(e)

# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Simple Image Upscaler & JPEG Converter with Color Profiles")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("--scale", type=int, default=None, help="Scale factor (e.g., 2, 3, 4). Omit for convert-only mode")
    parser.add_argument("--method", default="lanczos", choices=["lanczos", "bicubic", "nearest", "sips"], help="Upscaling method")
    parser.add_argument("--output", help="Output directory (default: input_dir/upscaled)")
    parser.add_argument("--replace", action="store_true", help="Replace original files")
    parser.add_argument("--no-suffix", action="store_true", help="Don't add _x{scale} suffix")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-one", action="store_true", help="Test with one image first")

    # New: Convert-only to JPEG w/ color profile
    parser.add_argument("--convert-jpg", action="store_true", help="Convert to JPEG")
    parser.add_argument("--color-profile", default="srgb", choices=["srgb", "adobe", "prophoto"], help="Color profile for JPEG output")
    parser.add_argument("--icc-dir", type=str, default=None, help="Directory containing ICC files (for Adobe/ProPhoto)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1–100)")
    parser.add_argument("--assume-input-srgb", action="store_true", help="Assume source is sRGB if no embedded ICC")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"❌ Error: Input directory '{input_dir}' not found")
        sys.exit(1)

    # Set output directory
    if args.replace:
        output_dir = input_dir
        print("🔄 Replace mode: Original files will be overwritten")
    elif args.output:
        output_dir = Path(args.output)
    else:
        output_sub = "output"
        if args.convert_jpg and args.scale is None:
            output_sub = "converted_jpg"
        elif args.convert_jpg and args.scale is not None:
            output_sub = f"upscaled_converted"
        else:
            output_sub = "upscaled"
        output_dir = input_dir / output_sub

    if not args.replace:
        output_dir.mkdir(parents=True, exist_ok=True)

    images = list(list_images(input_dir))

    if not images:
        print(f"❌ No images found in {input_dir}")
        print(f"Supported formats: {', '.join(VALID_EXT)}")
        sys.exit(1)

    print(f"📁 Found {len(images)} images to process")
    print(f"📂 Output directory: {output_dir}")

    # Determine mode
    convert_only = args.convert_jpg and (args.scale is None)
    do_upscale = args.scale is not None

    if convert_only:
        print("🎨 Mode: Convert-only to JPEG")
        print(f"🖨️  Target profile: {args.color_profile.upper()}")
        if args.color_profile.lower() in ["adobe", "prophoto"] and args.icc_dir is None:
            print("ℹ️  Tip: For Adobe/ProPhoto, consider providing --icc-dir to ensure correct ICC embedding.")
    elif do_upscale and args.convert_jpg:
        print("🔼 Mode: Upscale + Convert to JPEG")
        print(f"🔢 Scale factor: {args.scale}x")
        print(f"🎨 Upscale method: {args.method}")
        print(f"🖨️  Target profile: {args.color_profile.upper()}")
    elif do_upscale:
        print("🔼 Mode: Upscale only")
        print(f"🔢 Scale factor: {args.scale}x")
        print(f"🎨 Upscale method: {args.method}")
    else:
        print("❌ Nothing to do: specify --scale for upscaling and/or --convert-jpg for conversion.")
        sys.exit(1)

    if args.test_one:
        print("\n🧪 Test mode: Processing only the first image...")
        images = images[:1]

    if args.replace and not args.test_one:
        print("\n⚠️  WARNING: This will replace your original files!")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)

    success_count = 0
    failed_files = []
    icc_dir = Path(args.icc_dir) if args.icc_dir else None

    for img_path in tqdm(images, desc="Processing images"):
        try:
            # Determine pipeline per file
            if args.replace:
                temp_path = img_path.parent / f"temp_{img_path.stem}{img_path.suffix}"

                if do_upscale:
                    if args.method == "sips":
                        ok, msg = upscale_image_sips(img_path, temp_path, args.scale)
                    else:
                        ok, msg = upscale_image_pil(img_path, temp_path, args.scale, args.method)
                    if not ok:
                        if temp_path.exists(): temp_path.unlink(missing_ok=True)
                        failed_files.append((img_path.name, msg))
                        if args.verbose: print(f"❌ {img_path.name}: {msg}")
                        continue

                    # If also convert to jpg
                    if args.convert_jpg:
                        # Use a second temp
                        temp_jpg = img_path.parent / f"temp_{img_path.stem}.jpg"
                        ok, msg = convert_to_jpg_with_profile(temp_path, temp_jpg, args.color_profile, icc_dir, args.jpeg_quality, args.assume_input_srgb)
                        temp_path.unlink(missing_ok=True)
                        if ok:
                            img_path.unlink(missing_ok=True)
                            temp_jpg.rename(img_path.with_suffix(".jpg"))
                            success_count += 1
                            if args.verbose:
                                print(f"✅ {img_path.name} -> {img_path.with_suffix('.jpg').name}")
                        else:
                            temp_jpg.unlink(missing_ok=True)
                            failed_files.append((img_path.name, msg))
                            if args.verbose: print(f"❌ {img_path.name}: {msg}")
                    else:
                        # Replace original with upscaled same-format
                        img_path.unlink(missing_ok=True)
                        temp_path.rename(img_path)
                        success_count += 1
                        if args.verbose:
                            print(f"✅ {img_path.name} (replaced upscaled)")
                else:
                    # Convert-only
                    temp_jpg = img_path.parent / f"temp_{img_path.stem}.jpg"
                    ok, msg = convert_to_jpg_with_profile(img_path, temp_jpg, args.color_profile, icc_dir, args.jpeg_quality, args.assume_input_srgb)
                    if ok:
                        img_path.unlink(missing_ok=True)
                        temp_jpg.rename(img_path.with_suffix(".jpg"))
                        success_count += 1
                        if args.verbose:
                            print(f"✅ {img_path.name} -> {img_path.with_suffix('.jpg').name}")
                    else:
                        temp_jpg.unlink(missing_ok=True)
                        failed_files.append((img_path.name, msg))
                        if args.verbose:
                            print(f"❌ {img_path.name}: {msg}")

            else:
                # Non-replace: write to output_dir
                # Determine output target path
                if do_upscale:
                    if args.no_suffix:
                        up_path = output_dir / img_path.name
                    else:
                        up_path = output_dir / f"{img_path.stem}_x{args.scale}{img_path.suffix}"

                    if args.method == "sips":
                        ok, msg = upscale_image_sips(img_path, up_path, args.scale)
                    else:
                        ok, msg = upscale_image_pil(img_path, up_path, args.scale, args.method)

                    if not ok:
                        failed_files.append((img_path.name, msg))
                        if args.verbose:
                            print(f"❌ {img_path.name}: {msg}")
                        continue

                    if args.convert_jpg:
                        jpg_path = up_path.with_suffix(".jpg")
                        ok, msg = convert_to_jpg_with_profile(up_path, jpg_path, args.color_profile, icc_dir, args.jpeg_quality, args.assume_input_srgb)
                        if ok:
                            success_count += 1
                            if args.verbose:
                                print(f"✅ {img_path.name} -> {jpg_path.name}")
                        else:
                            failed_files.append((img_path.name, f"Conversion error: {msg}"))
                            if args.verbose:
                                print(f"❌ {img_path.name}: {msg}")
                    else:
                        success_count += 1
                        if args.verbose:
                            print(f"✅ {img_path.name} -> {up_path.name}")

                else:
                    # Convert-only
                    jpg_path = output_dir / f"{img_path.stem}.jpg"
                    ok, msg = convert_to_jpg_with_profile(img_path, jpg_path, args.color_profile, icc_dir, args.jpeg_quality, args.assume_input_srgb)
                    if ok:
                        success_count += 1
                        if args.verbose:
                            print(f"✅ {img_path.name} -> {jpg_path.name}")
                    else:
                        failed_files.append((img_path.name, msg))
                        if args.verbose:
                            print(f"❌ {img_path.name}: {msg}")

        except Exception as e:
            failed_files.append((img_path.name, str(e)))
            if args.verbose:
                print(f"❌ Error: {img_path.name}: {e}")

    mode_label = "Completed"
    print(f"\n🎉 {mode_label}! {success_count}/{len(images)} images processed successfully")

    if failed_files:
        print(f"\n❌ Failed files ({len(failed_files)}):")
        for filename, error in failed_files[:5]:
            print(f"  • {filename}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    if not args.replace and success_count > 0:
        print(f"\n📁 Output saved to: {output_dir}")

    if args.test_one and success_count > 0:
        print("\n✅ Test successful! Run without --test-one to process all images.")

if __name__ == "__main__":
    main()
