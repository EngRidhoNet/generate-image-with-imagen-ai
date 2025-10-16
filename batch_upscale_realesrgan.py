import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageFilter
from tqdm import tqdm

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def list_images(root: Path):
    """Find all image files in directory"""
    for p in root.rglob("*"):
        if p.suffix.lower() in VALID_EXT and p.is_file():
            yield p

def upscale_image_pil(input_path: Path, output_path: Path, scale: int, method="lanczos"):
    """Upscale image using PIL with high-quality resampling"""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if needed (for compatibility)
            if img.mode not in ['RGB', 'RGBA']:
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            
            # Get original size
            width, height = img.size
            new_width = width * scale
            new_height = height * scale
            
            # Choose resampling method
            if method == "lanczos":
                resample = Image.LANCZOS
            elif method == "bicubic":
                resample = Image.BICUBIC
            elif method == "nearest":
                resample = Image.NEAREST
            else:
                resample = Image.LANCZOS
            
            # Upscale
            upscaled = img.resize((new_width, new_height), resample)
            
            # Apply slight sharpening for better quality
            if method == "lanczos":
                upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=0))
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high quality
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
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
        
        # Get original dimensions
        with Image.open(input_path) as img:
            width, height = img.size
        
        new_width = width * scale
        new_height = height * scale
        
        # Ensure output directory exists
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

def main():
    parser = argparse.ArgumentParser(description="Simple Image Upscaler")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="Scale factor")
    parser.add_argument("--method", default="lanczos", 
                       choices=["lanczos", "bicubic", "sips"], help="Upscaling method")
    parser.add_argument("--output", help="Output directory (default: input_dir/upscaled)")
    parser.add_argument("--replace", action="store_true", help="Replace original files")
    parser.add_argument("--no-suffix", action="store_true", help="Don't add _x{scale} suffix")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-one", action="store_true", help="Test with one image first")
    
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
        output_dir = input_dir / "upscaled"
    
    if not args.replace:
        output_dir.mkdir(exist_ok=True)
    
    # Find all images
    images = list(list_images(input_dir))
    
    if not images:
        print(f"❌ No images found in {input_dir}")
        print(f"Supported formats: {', '.join(VALID_EXT)}")
        sys.exit(1)
    
    print(f"📁 Found {len(images)} images to process")
    print(f"📂 Output directory: {output_dir}")
    print(f"🔢 Scale factor: {args.scale}x")
    print(f"🎨 Method: {args.method}")
    
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
    
    for img_path in tqdm(images, desc="Processing images"):
        try:
            if args.replace:
                # Create temporary file first
                temp_path = img_path.parent / f"temp_upscaled_{img_path.name}"
                
                if args.method == "sips":
                    success, message = upscale_image_sips(img_path, temp_path, args.scale)
                else:
                    success, message = upscale_image_pil(img_path, temp_path, args.scale, args.method)
                
                if success:
                    # Replace original
                    img_path.unlink()
                    temp_path.rename(img_path)
                    success_count += 1
                    if args.verbose:
                        print(f"✅ {img_path.name}")
                else:
                    if temp_path.exists():
                        temp_path.unlink()
                    failed_files.append((img_path.name, message))
                    if args.verbose:
                        print(f"❌ {img_path.name}: {message}")
            else:
                # Create new file
                if args.no_suffix:
                    output_path = output_dir / img_path.name
                else:
                    output_path = output_dir / f"{img_path.stem}_x{args.scale}{img_path.suffix}"
                
                if args.method == "sips":
                    success, message = upscale_image_sips(img_path, output_path, args.scale)
                else:
                    success, message = upscale_image_pil(img_path, output_path, args.scale, args.method)
                
                if success:
                    success_count += 1
                    if args.verbose:
                        print(f"✅ {img_path.name} -> {output_path.name}")
                else:
                    failed_files.append((img_path.name, message))
                    if args.verbose:
                        print(f"❌ {img_path.name}: {message}")
                    
        except Exception as e:
            failed_files.append((img_path.name, str(e)))
            if args.verbose:
                print(f"❌ Error: {img_path.name}: {e}")
    
    print(f"\n🎉 Completed! {success_count}/{len(images)} images processed successfully")
    
    if failed_files:
        print(f"\n❌ Failed files ({len(failed_files)}):")
        for filename, error in failed_files[:5]:
            print(f"  • {filename}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    if not args.replace and success_count > 0:
        print(f"\n📁 Upscaled images saved to: {output_dir}")
    
    if args.test_one and success_count > 0:
        print("\n✅ Test successful! Run without --test-one to process all images.")

if __name__ == "__main__":
    main()