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
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument("--test-split", type=float, default=0.15, help="Fraction of data for testing")
    parser.add_argument("--model-out", default="best_model.keras", help="Where to save the trained model")
    parser.add_argument("--plot-out", default=None, help="Optional path to save training plot")
    return parser.parse_args()


def main():
    args = parse_args()
    df = prepare_dataframe(args.csv)
    
    train_gen, val_gen, test_gen = create_generators(
        df,
        args.images,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    model = build_model()
    history = train_model(
        model,
        train_gen,
        val_gen,
        epochs=args.epochs,
        output_path=args.model_out,
    )
    plot_history(history, output=args.plot_out)

    test_metrics = model.evaluate(
        test_gen,
        steps=test_gen.n // test_gen.batch_size,
        verbose=1,
    )
    print(
        f"Test loss: {test_metrics[0]:.4f} - mse: {test_metrics[1]:.4f} - "
        f"accuracy: {test_metrics[2]:.4f}"
    )

if __name__ == "__main__":
    main()
