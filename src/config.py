from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "alzheimers_disease_data.csv"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)