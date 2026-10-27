from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = ROOT / "models" / "turnover_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "Attrition"
ID_COL = "EmployeeNumber"
# Constant columns in this dataset carry no signal.
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours"]
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
HIGH_RISK = 0.5
MEDIUM_RISK = 0.25
