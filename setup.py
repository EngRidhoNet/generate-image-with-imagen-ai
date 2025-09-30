from setuptools import setup

setup(
    name="upscale",
    version="0.1.0",
    py_modules=["batch_upscale_realesrgan"],  # pakai py_modules karena 1 file
    install_requires=[
        # biarkan lib besar (torch/vision/basicsr) DIKELOLA via conda/pip manual per-env
        "opencv-python",
        "pillow",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "upscale=batch_upscale_realesrgan:main",
        ],
    },
)
