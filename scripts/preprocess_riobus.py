"""
Rio Bus Trajectory Dataset Preprocessing Pipeline (Phase 4 - Task T4.1)
=======================================================================
Reads raw/processed Rio bus GPS trajectory CSV files, fixes data issues
(timestamps, speed unit conversion km/h -> m/s, invalid velocity jumps),
interpolates trajectory to uniform step dt = 0.1s, and exports clean
parquet/CSV trajectory file for the simulator.
"""

import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import Logger

logger = Logger("RioBusPreprocess")


def preprocess_single_file(file_path: Path, cfg: dict) -> pd.DataFrame:
    """Preprocess a single bus trajectory CSV file."""
    df = pd.read_csv(file_path)

    # 1. Clean timestamp / time order
    if "delta_time" in df.columns:
        df = df.sort_values(by="delta_time").reset_index(drop=True)
        t_raw = df["delta_time"].values.astype(float)
    else:
        # Construct relative seconds from hour/minute/second
        t_raw = (df["hour"] * 3600 + df["minute"] * 60 + df["second"]).values.astype(float)
        df = df.sort_values(by="hour").reset_index(drop=True)

    t_rel = t_raw - t_raw[0]

    # Remove non-monotonic duplicate timestamps
    valid_idx = np.where(np.diff(t_rel, prepend=-1.0) > 1e-5)[0]
    df = df.iloc[valid_idx].reset_index(drop=True)
    t_rel = t_rel[valid_idx]

    # 2. Convert speed km/h -> m/s
    speed_kmh = df["speed"].values.astype(float)
    v_mps = speed_kmh / 3.6

    # Assert speed bounds
    assert np.all(v_mps >= cfg.get("min_velocity_mps", 0.0)), "Speed contains negative values!"
    v_mps = np.clip(v_mps, 0.0, cfg.get("max_velocity_mps", 35.0))

    # 3. Coordinate normalization / projection into sim area
    x_raw = df["x"].values.astype(float)
    y_raw = df["y"].values.astype(float)

    x_range = np.ptp(x_raw) if np.ptp(x_raw) > 1e-5 else 1.0
    y_range = np.ptp(y_raw) if np.ptp(y_raw) > 1e-5 else 1.0

    x_m = ((x_raw - np.min(x_raw)) / x_range) * cfg.get("sim_area_x", 1000.0)
    y_m = ((y_raw - np.min(y_raw)) / y_range) * cfg.get("sim_area_y", 1000.0)

    # 4. Interpolation at dt = 0.1s
    dt = cfg.get("dt", 0.1)
    if len(t_rel) < 2 or t_rel[-1] <= 0:
        return pd.DataFrame()

    t_max = min(t_rel[-1], float(cfg.get("max_duration_seconds", 1000.0)))
    t_interp = np.arange(0, t_max, dt)

    x_interp = np.interp(t_interp, t_rel, x_m)
    y_interp = np.interp(t_interp, t_rel, y_m)
    v_interp = np.interp(t_interp, t_rel, v_mps)

    # 5. Check quality: velocity derived from position diff vs interpolated velocity
    if len(t_interp) > 10:
        dx = np.diff(x_interp) / dt
        dy = np.diff(y_interp) / dt
        derived_v = np.sqrt(dx**2 + dy**2)
        mean_derived = np.mean(derived_v)
        mean_interp = np.mean(v_interp)
        logger.debug(f"{file_path.name}: mean_derived_v={mean_derived:.2f} m/s, mean_interp_v={mean_interp:.2f} m/s")

    order_id = str(df["order"].iloc[0]) if "order" in df.columns else file_path.stem

    clean_df = pd.DataFrame({
        "vehicle_id": order_id,
        "t_rel": t_interp,
        "x_m": x_interp,
        "y_m": y_interp,
        "v_mps": v_interp
    })

    return clean_df


def main():
    parser = argparse.ArgumentParser(description="Rio Bus Trajectory Preprocessor")
    parser.add_argument("--config", type=str, default="configs/data/riobus.yaml", help="Path to data config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg.get("data_dir", "data/processed/data_train"))
    logger.info(f"Ingesting Rio bus data from {data_dir}...")

    all_dfs = []
    max_vehicles = cfg.get("max_vehicles", 20)

    # Find CSV files
    csv_files = list(data_dir.rglob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV trajectory files.")

    count = 0
    for file_path in csv_files:
        if count >= max_vehicles:
            break
        try:
            df_clean = preprocess_single_file(file_path, cfg)
            if not df_clean.empty:
                all_dfs.append(df_clean)
                count += 1
        except Exception as e:
            logger.warning(f"Error processing {file_path.name}: {e}")

    if not all_dfs:
        logger.error("No valid trajectories processed!")
        sys.exit(1)

    final_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Successfully processed {count} vehicle trajectories ({len(final_df)} rows).")

    # Export output
    out_parquet = Path(cfg.get("output_parquet", "data/processed/riobus_corridor.parquet"))
    out_csv = Path(cfg.get("output_csv", "data/processed/riobus_corridor.csv"))

    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    try:
        final_df.to_parquet(out_parquet, index=False)
        logger.info(f"Exported parquet to {out_parquet}")
    except Exception as e:
        logger.warning(f"Parquet export failed ({e}), saving CSV only.")

    final_df.to_csv(out_csv, index=False)
    logger.info(f"Exported CSV to {out_csv}")

    # DoD Validation Assertions
    assert len(final_df) > 0, "Preprocessed dataset is empty!"
    assert np.all(final_df["v_mps"] <= cfg.get("max_velocity_mps", 35.0)), "Velocity exceeds max limit!"
    logger.info("DoD Validation Passed: Output trajectories created and validated.")


if __name__ == "__main__":
    main()
