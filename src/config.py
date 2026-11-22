from pathlib import Path

# Project root = the folder that contains src/
ROOT = Path(__file__).resolve().parents[1]

# Where things live
DATA_RAW = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# Dataset settings
CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
SEED = 42
IMG_SIZE = 224

# Create output folders automatically if they don't exist
for d in (SPLITS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)