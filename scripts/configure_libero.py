"""Write a local LIBERO config before importing the external simulator package."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero_dir", type=Path, default=Path("third_party/LIBERO"))
    parser.add_argument("--config_dir", type=Path, default=Path(".cache/libero"))
    args = parser.parse_args()
    root = args.libero_dir.expanduser().resolve() / "libero/libero"
    if not (root / "bddl_files").is_dir():
        parser.error(f"LIBERO checkout not found at {root.parent.parent}")
    config = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "assets": str(root / "assets"),
        "datasets": str(Path(__file__).resolve().parents[1] / "data/LIBERO_datasets"),
    }
    config_dir = args.config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML; writing before import avoids upstream interactive setup.
    path = config_dir / "config.yaml"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {path}")
    print(f"Set LIBERO_CONFIG_PATH to {config_dir} before training/evaluation.")


if __name__ == "__main__":
    main()
