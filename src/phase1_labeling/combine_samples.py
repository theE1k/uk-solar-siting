"""Combine positive and negative samples into a single labelled dataset."""

import pandas as pd

from src.utils import load_config, resolve_path
from src.phase1_labeling.generate_positive import generate_positive
from src.phase1_labeling.generate_negative import generate_negative


def combine_samples(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    positives = generate_positive(cfg)
    negatives = generate_negative(positives, cfg)

    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined.sample(frac=1, random_state=cfg["labeling"]["random_seed"]).reset_index(drop=True)

    out_path = resolve_path(cfg["paths"]["processed_data"]) / "labeled_sites.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"[Phase1] Combined dataset: {len(combined)} rows ({positives.shape[0]} pos / {negatives.shape[0]} neg)")
    print(f"[Phase1] Saved to {out_path}")
    return combined


if __name__ == "__main__":
    df = combine_samples()
    print(df.head(10))
