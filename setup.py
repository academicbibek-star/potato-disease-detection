import subprocess
import sys


def install(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


for p in [
    "timm", "albumentations", "imagehash", "scikit-learn", "pandas",
    "matplotlib", "seaborn", "opencv-python-headless", "scipy"
]:
    install(p)

print("All dependencies installed.")
