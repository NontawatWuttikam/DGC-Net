import pandas as pd
from pathlib import Path
import argparse
import yaml
import itertools

def main():
    parser = argparse.ArgumentParser(description="Generate CSV with matched target image paths.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--proxydgc-config", type=str, required=True, help="Path to the proxy-dgc-net config file.")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for sampling.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    yaml_dict = yaml.safe_load(open(args.proxydgc_config, "r"))
    raw_dir = Path(yaml_dict["target_images"])
    raw_dir_name = raw_dir.name  # e.g. "tinyTimeMachine"

    # Load original CSV
    df = pd.read_csv(csv_path)

    # Find all .dng files
    target_image_paths = list(raw_dir.rglob("*.dng"))
    if len(target_image_paths) == 0:
        raise ValueError(f"No .dng files found in {raw_dir}")

    # Filter only "homo" entries
    df = df[df["aff/tps/homo"] == 2].reset_index(drop=True)

    # Repeat the image paths to match the number of rows in the filtered DataFrame
    repeated_paths = list(itertools.islice(itertools.cycle(target_image_paths), len(df)))

    # Update fname column with relative paths
    df["fname"] = [str(p).split("data/")[-1] for p in repeated_paths]

    # Construct output filename using raw dir name
    output_csv = csv_path.with_name(f"homo_aff_tps_train_{raw_dir_name}.csv")

    # Save new CSV
    df.to_csv(output_csv, index=False, sep=",")

    # Print path to stdout
    print(str(output_csv))

if __name__ == "__main__":
    main()
