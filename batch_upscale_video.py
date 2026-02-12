import sys
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

VALID_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

def list_videos(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in VALID_VIDEO_EXT and p.is_file():
            yield p

def upscale_video_ffmpeg(input_path: Path, output_path: Path, scale: int):
    """
    Upscale video using FFmpeg with Lanczos filter
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vf", f"scale=iw*{scale}:ih*{scale}:flags=lanczos",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(output_path)
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True, "Success"

    except subprocess.CalledProcessError as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Batch Video Upscaler (FFmpeg Lanczos)")
    parser.add_argument("input_dir", help="Folder containing videos")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="Upscale factor")
    parser.add_argument("--output", help="Output directory (default: input_dir/upscaled_videos)")
    parser.add_argument("--replace", action="store_true", help="Replace original files")
    parser.add_argument("--no-suffix", action="store_true", help="Do not add _x{scale} suffix")
    parser.add_argument("--test-one", action="store_true", help="Test only one video")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"❌ Folder tidak ditemukan: {input_dir}")
        sys.exit(1)

    if args.replace:
        output_dir = input_dir
        print("⚠️ Replace mode: video asli akan ditimpa")
    elif args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir / "upscaled_videos"

    if not args.replace:
        output_dir.mkdir(exist_ok=True)

    videos = list(list_videos(input_dir))

    if not videos:
        print("❌ Tidak ada video ditemukan")
        print("Format didukung:", ", ".join(VALID_VIDEO_EXT))
        sys.exit(1)

    print(f"🎬 Total video: {len(videos)}")
    print(f"📈 Scale: {args.scale}x")
    print(f"📂 Output: {output_dir}")

    if args.test_one:
        print("🧪 Test mode aktif (1 video)")
        videos = videos[:1]

    if args.replace and not args.test_one:
        confirm = input("⚠️ Yakin replace semua video? (y/N): ")
        if confirm.lower() != "y":
            print("Dibatalkan.")
            sys.exit(0)

    success = 0
    failed = []

    for video in tqdm(videos, desc="Upscaling videos"):
        try:
            if args.replace:
                temp_path = video.parent / f"__temp_{video.name}"
                ok, msg = upscale_video_ffmpeg(video, temp_path, args.scale)
                if ok:
                    video.unlink()
                    temp_path.rename(video)
                    success += 1
                else:
                    failed.append((video.name, msg))
            else:
                if args.no_suffix:
                    output_path = output_dir / video.name
                else:
                    output_path = output_dir / f"{video.stem}_x{args.scale}{video.suffix}"

                ok, msg = upscale_video_ffmpeg(video, output_path, args.scale)
                if ok:
                    success += 1
                else:
                    failed.append((video.name, msg))

            if args.verbose:
                print(f"✅ {video.name}")

        except Exception as e:
            failed.append((video.name, str(e)))

    print(f"\n🎉 Selesai: {success}/{len(videos)} video berhasil")

    if failed:
        print("\n❌ Gagal:")
        for f, err in failed[:5]:
            print(f"• {f} → {err}")

    if args.test_one and success:
        print("\n✅ Test OK. Jalankan tanpa --test-one untuk batch penuh.")


if __name__ == "__main__":
    main()
