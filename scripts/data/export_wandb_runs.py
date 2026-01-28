"""Export wandb runs to CSV.

Usage:
    uv run scripts/data/export_wandb_runs.py \
        --entity "your-wandb-entity" \
        --project "your-project-name" \
        --output path/to/output.csv
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")
from src.utils.wandb import export_runs_to_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Export wandb runs to CSV")
    parser.add_argument("--entity", type=str, required=True, help="wandb entity (username or team)")
    parser.add_argument("--project", type=str, required=True, help="wandb project name")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: data/wandb/<project>_runs.csv)",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "wandb"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{args.project}_runs.csv"
    else:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting runs from {args.entity}/{args.project}")
    print(f"Output file: {output_file}")

    export_runs_to_csv(args.entity, args.project, str(output_file))


if __name__ == "__main__":
    main()
