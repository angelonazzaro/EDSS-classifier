import argparse
import csv
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm

from edss_dataset import get_dataset
from model.cnn import CNNModel
from model.vit import VisionTransformer
from utils import parse_tuple
from utils.constants import CLASS_THRESHOLDS


def load_model(checkpoint_paths, model_types):
    """
    Loads model from checkpoint path.
    If more than one checkpoint is passed, an ensemble model is build with majority voting.

    Args:
        - checkpoint_paths (list): List of checkpoint paths.
        - model_types (list): List of model types (either CNN or ViT).
            Must match the same number of elements of checkpoint_paths. If one is passed, it is repeated.
    """

    if isinstance(model_types, str):
        model_types = [model_types]

    if len(model_types) == 1:
        model_types = [model_types[0]] * len(checkpoint_paths)

    if len(model_types) != len(checkpoint_paths):
        raise ValueError(
            "`model_types` must have the same number of elements as `checkpoint_paths`"
        )

    models = [
        tf.keras.models.load_model(
            ck,
            custom_objects={"CNNModel": CNNModel}
            if model_types[i] == "CNN"
            else {"VisionTransformer": VisionTransformer},
        )
        for i, ck in enumerate(checkpoint_paths)
    ]

    def predict(x):
        preds = [m.predict(x) for m in models]
        preds = np.array(preds)  # shape: (n_models, batch, classes)

        # Majority voting
        votes = np.argmax(preds, axis=-1)  # shape: (n_models, batch)
        maj_vote = [np.bincount(col).argmax() for col in votes.T]
        return np.array(maj_vote)

    return predict

def test(args):
    # ensure reproducible results
    tf.keras.utils.set_random_seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    os.makedirs(args.results_dir, exist_ok=True)
    model_results_dir = os.path.join(args.results_dir, args.task, args.model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    csv_path = os.path.join(args.results_dir, f"{args.task}_results.csv")

    test_dataset = get_dataset(
        data_dir=args.data_dir,
        split="test",
        modality=args.modality,
        task=args.task,
        batch_size=args.batch_size,
        resize=args.resize,
    )

    if isinstance(args.modality, List):
        args.modality = " ".join(args.modality)
    
    model = load_model(args.checkpoint_path, args.model_type)

    write_header = not os.path.exists(csv_path)

    labels = list(CLASS_THRESHOLDS[args.task].keys())

    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(
                ["model_name", "modality", "accuracy", "precision", "recall", "f1"]
            )

        print(
            f"\n=== Testing {args.model_name} on {args.task} classification ({args.modality}) ==="
        )

        y_true = []
        y_pred = []

        for x_batch, y_batch in tqdm(
            test_dataset, desc="Testing", total=len(test_dataset)
        ):
            preds = model(x_batch)  # noqa

            if args.task == "binary":
                y_true.extend(y_batch.numpy().astype(int))
            else:
                y_true.extend(np.argmax(y_batch.numpy(), axis=1))

            y_pred.extend(preds)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true,
            y_pred,
            average=None if args.task == "binary" else "micro",
            zero_division=0,
        )
        recall = recall_score(
            y_true,
            y_pred,
            average=None if args.task == "binary" else "micro",
            zero_division=0,
        )
        f1 = f1_score(
            y_true,
            y_pred,
            average=None if args.task == "binary" else "micro",
            zero_division=0,
        )

        writer.writerow(
            [args.model_name, args.modality, accuracy, precision, recall, f1]
        )

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=labels, cmap="Blues"
        )
        disp.ax_.set_title(
            f"Confusion Matrix {args.task} classification\n{args.model_name} - {args.modality}"
        )
        disp.figure_.savefig(os.path.join(model_results_dir, f"cm_{args.modality}.png"))  # noqa
        plt.close(disp.figure_)

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random generation"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save model results",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        nargs='+',
        help="Path(s) of the model checkpoint(s) to test. "
        "If more than one is passed, an ensemble is built with majority voting.",
    )

    parser.add_argument("--data_dir", required=True, type=str, help="Data directory")
    parser.add_argument(
        "--modality",
        type=str,
        choices=["T1", "T2", "FLAIR"],
        nargs="+",
        default="T1",
        help="MRI Modality",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binary", "multi-class"],
        default="binary",
        help="Task to train the model onto",
    )
    parser.add_argument(
        "--resize",
        type=parse_tuple,
        help="Tuple of height and width to which resize the images, e.g. 256,256 or (256,256)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument(
        "--model_type", type=str, nargs="+", choices=["CNN", "ViT"], default="CNN"
    )

    args = parser.parse_args()

    test(args)

