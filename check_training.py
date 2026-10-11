
import os

# Search project Python files for the model training code
for root, dirs, files in os.walk("."):
    dirs[:] = [
        d for d in dirs
        if d not in {".venv", "__pycache__", ".git"}
    ]

    for filename in files:
        if filename.endswith(".py") and filename not in {
            "check_model.py",
            "check_training.py",
            "streamlit_app.py",
        }:
            path = os.path.join(root, filename)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                if any(term in content.lower() for term in [
                    "xgbregressor",
                    "feature_names",
                    "molecularweight",
                    "save_model",
                ]):
                    print(f"\n{'=' * 60}")
                    print(f"FILE: {path}")
                    print("=" * 60)

                    for i, line in enumerate(content.splitlines(), 1):
                        if any(term in line.lower() for term in [
                            "xgbregressor",
                            "feature_names",
                            "molecularweight",
                            "logp",
                            "tpsa",
                            "fit(",
                            "save_model",
                            "joblib.dump",
                        ]):
                            print(f"{i}: {line}")

            except (UnicodeDecodeError, OSError):
                pass
