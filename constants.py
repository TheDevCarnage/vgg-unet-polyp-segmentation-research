RAW_DATA_DIR = "data/raw"
DATASET_NAME = "balraj98/cvcclinicdb"
ZIP_FILENAME = "cvcclinicdb.zip"
EXPECTED_IMAGES = 612
IMAGE_FOLDER_NAME = "Original"
MASK_FOLDER_NAME = "Ground Truth"

# Split ratios — must sum to 1.0
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Supported formatts
SUPPORTED_FORMATS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


IMAGE_DIR_TIF = f"{RAW_DATA_DIR}/TIF/{IMAGE_FOLDER_NAME}"
MASK_DIR_TIF = f"{RAW_DATA_DIR}/TIF/{MASK_FOLDER_NAME}"

IMAGE_DIR_PNG = f"{RAW_DATA_DIR}/PNG/{IMAGE_FOLDER_NAME}"
MASK_DIR_PNG = f"{RAW_DATA_DIR}/PNG/{MASK_FOLDER_NAME}"

OUTPUT_DIR = "data/processed"
TRAIN_IMG_DIR = "data/processed/train/images"
TRAIN_MASK_DIR = "data/processed/train/masks"
VAL_IMG_DIR = "data/processed/val/images"
VAL_MASK_DIR = "data/processed/val/masks"
TEST_IMG_DIR = "data/processed/test/images"
TEST_MASK_DIR = "data/processed/test/masks"

# ── Training Hyperparameters ──────────────────────────
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Save resulting figures here
RESULTING_FIGURES_DIR = "results/figures"

CHECKPOINT_DIR = "results/checkpoints"
