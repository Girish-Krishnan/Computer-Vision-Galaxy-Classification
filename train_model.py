#!/usr/bin/env python3
"""Train a galaxy classification model."""
import argparse
from galaxy.data_utils import prepare_dataframe, create_generators
from galaxy.model_utils import build_model, train_model, plot_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train the galaxy classification model")
    parser.add_argument("--csv", default="data/training_solutions_rev1.csv", help="Path to labels CSV")
    parser.add_argument("--images", default="data/images_training_rev1/", help="Directory with training images")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-out", default="best_model.keras", help="Where to save the trained model")
    parser.add_argument("--plot-out", default=None, help="Optional path to save training plot")
    return parser.parse_args()


def main():
    args = parse_args()
    df = prepare_dataframe(args.csv)
    train_gen, val_gen = create_generators(df, args.images, batch_size=args.batch_size)
    model = build_model()
    history = train_model(model, train_gen, val_gen, epochs=args.epochs, output_path=args.model_out)
    plot_history(history, output=args.plot_out)


if __name__ == "__main__":
    main()
