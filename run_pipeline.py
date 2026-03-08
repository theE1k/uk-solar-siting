#!/usr/bin/env python3
"""
UK Solar Siting Pipeline — End-to-end runner.

Usage:
    python run_pipeline.py                  # run all phases
    python run_pipeline.py --phase 1        # run only phase 1
    python run_pipeline.py --phase 2 3      # run phases 2 and 3
"""

import argparse
import sys
import time

from src.utils import load_config


def run_phase1(cfg):
    print("\n" + "=" * 60)
    print("PHASE 1 — Generate Labelled Data")
    print("=" * 60)
    from src.phase1_labeling.download_repd import download_repd
    from src.phase1_labeling.combine_samples import combine_samples

    download_repd(cfg)
    combine_samples(cfg)


def run_phase2(cfg):
    print("\n" + "=" * 60)
    print("PHASE 2 — Feature Engineering")
    print("=" * 60)
    from src.phase2_features.build_feature_matrix import build_feature_matrix

    build_feature_matrix(cfg)


def run_phase3(cfg):
    print("\n" + "=" * 60)
    print("PHASE 3 — Model Training & Evaluation")
    print("=" * 60)
    from src.phase3_model.train import train_model
    from src.phase3_model.evaluate import evaluate_model

    train_model(cfg)
    evaluate_model(cfg)


def run_phase4(cfg):
    print("\n" + "=" * 60)
    print("PHASE 4 — Grid Prediction & Heatmap")
    print("=" * 60)
    from src.phase4_prediction.predict import predict_grid
    from src.phase4_prediction.heatmap import generate_heatmap

    predict_grid(cfg)
    generate_heatmap(cfg)


PHASES = {1: run_phase1, 2: run_phase2, 3: run_phase3, 4: run_phase4}


def main():
    parser = argparse.ArgumentParser(description="UK Solar Siting Pipeline")
    parser.add_argument("--phase", nargs="*", type=int, default=None,
                        help="Phase number(s) to run. Default: all.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    phases_to_run = args.phase if args.phase else sorted(PHASES.keys())

    print("UK Solar Siting Pipeline")
    print(f"Phases to run: {phases_to_run}")
    t0 = time.time()

    for phase_num in phases_to_run:
        if phase_num not in PHASES:
            print(f"ERROR: Unknown phase {phase_num}. Valid: {list(PHASES.keys())}")
            sys.exit(1)
        PHASES[phase_num](cfg)

    elapsed = time.time() - t0
    print(f"\nPipeline complete. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
