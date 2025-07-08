#!/usr/bin/env python3
"""Extract features using a trained model."""
import argparse
from galaxy.feature_extraction import extract_features


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from images")
    parser.add_argument("--model", default="best_model.keras", help="Path to trained model")
    parser.add_argument("--csv", default="data/training_solutions_rev1.csv", help="Path to labels CSV")
    parser.add_argument("--images", default="data/images_training_rev1/", help="Directory with images")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--out", default="features.npy", help="Output .npy file for features")
    return parser.parse_args()


def main():
    args = parse_args()
    extract_features(args.model, args.csv, args.images,
                     batch_size=args.batch_size, output_path=args.out)


if __name__ == "__main__":
    main()
