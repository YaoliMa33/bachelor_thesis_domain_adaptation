"""Train the independent seed pool for MSPA using the shared UCI trainer."""
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parent.parent / "dann_cdan_lc"
sys.path.insert(0, str(SHARED))
from main import main as train_members


if __name__ == "__main__":
    if not any(a == "--output" or a.startswith("--output=") for a in sys.argv[1:]):
        sys.argv.extend(["--output", str(Path(__file__).resolve().parent / "results")])
    train_members()
