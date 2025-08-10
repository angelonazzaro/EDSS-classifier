import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from edss_dataset import get_dataset
from model.cnn import CNNModel
from model.vit import VisionTransformer
from utils.constants import CLASS_THRESHOLDS


def test(args):

    # ensure reproducible results
    tf.keras.utils.set_random_seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    os.makedirs(args.results_dir, exist_ok=True)
    model_results_dir = os.path.join(args.results_dir, args.task)
    model_results_dir = os.path.join(model_results_dir, args.model_name)  # noqa
    os.makedirs(model_results_dir, exist_ok=True)

    csv_path = os.path.join(args.results_dir, f"{args.task}_results.csv")

    test_dataset = get_dataset(data_dir=args.data_dir, split="test",
                               modality=args.modality, task=args.task,
                               batch_size=args.batch_size, resize=args.resize)

    model = tf.keras.models.load_model(args.checkpoint_path,
                                       custom_objects={"CNNModel": CNNModel} if args.model_type == "CNN" else {
                                           "VisionTransformer": VisionTransformer})
    write_header = not os.path.exists(csv_path)

    labels = list(CLASS_THRESHOLDS[args.task].keys())

    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["model_name", "modality", "accuracy", "precision", "recall", "f1"])

        print(f"\n=== Testing {args.model_name} on {args.task} classification ({args.modality}) ===")

        y_true = []
        y_pred = []

        # Run inference
        for x_batch, y_batch in tqdm(test_dataset, desc="Testing", total=len(test_dataset)):
            preds = model.predict(x_batch)
            if preds.shape[1] == 1:
                preds_class = (preds > 0.5).astype(int)
                y_true.extend(y_batch.numpy().astype(int))
                y_pred.extend(preds_class)
            else:
                preds_class = np.argmax(preds, axis=1)
                y_true.extend(np.argmax(y_batch.numpy(), axis=1))
                y_pred.extend(preds_class)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        writer.writerow([args.model_name, args.modality, accuracy, precision, recall, f1])

        # Confusion matrix
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues')
        disp.ax_.set_title(f"Confusion Matrix {args.task} classification\n{args.model_name} - {args.modality}")
        disp.figure_.savefig(os.path.join(model_results_dir, f"cm_{args.modality}.png"))  # noqa
        plt.close(disp.figure_)

    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    def parse_tuple(string):
        try:
            items = string.strip("()").split(",")
            return tuple(int(i) for i in items)
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be numbers separated by commas")


    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Seed for random generation')

    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save model results')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path of the model checkpoint to test.')
    parser.add_argument('--weights_only', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to load weights only')

    parser.add_argument('--data_dir', required=True, type=str, help='Data directory')
    parser.add_argument('--modality', type=str, choices=['T1', 'T2', 'FLAIR'], nargs='+', default='T1', help='MRI Modality')
    parser.add_argument('--task', type=str, choices=['binary', 'multi-class'], default='binary',
                        help='Task to train the model onto')
    parser.add_argument('--resize', type=parse_tuple,
                        help="Tuple of height and width to which resize the images, e.g. 256,256 or (256,256)")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument("--model_type", type=str, choices=['CNN', 'ViT'], default='CNN')

    args = parser.parse_args()

    test(args)