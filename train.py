import argparse
import logging
import os

import tensorflow as tf
import wandb
from tqdm import tqdm

from edss_dataset import get_dataset
from model.cnn import CNNModel
from utils.early_monitoring import EarlyCheckpointing

logger = logging.getLogger(__name__)


def run_epoch(model, dataset, criterion, task, epoch, epochs=None, optimizer=None, training=False):
    """Run one epoch of training or validation."""
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.BinaryAccuracy() if task == "binary" else tf.keras.metrics.CategoricalAccuracy()
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()

    desc = f"{'Training' if training else 'Validation'} Epoch {epoch}" + (f"/{epochs}" if training else "")

    for x_batch, y_batch in tqdm(dataset, desc=desc):
        if training:
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss_value = criterion(y_batch, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            logits = model(x_batch, training=False)
            loss_value = criterion(y_batch, logits)

        loss_metric.update_state(loss_value)
        acc_metric.update_state(y_batch, logits)
        precision_metric.update_state(y_batch, logits)
        recall_metric.update_state(y_batch, logits)

    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))

    metrics = {
        "loss": loss_metric.result().numpy(),
        "acc": acc_metric.result().numpy(),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    prefix = "train_" if training else "val_"
    wandb.log({f"{prefix}{k}": v for k, v in metrics.items()} | {"epoch": epoch})

    return metrics


def main(args):
    wandb.init(project=args.project, config=dict(vars(args)), name=args.experiment_name)

    if args.experiment_name is None:
        args.experiment_name = wandb.run.name

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)

    train_dataset = get_dataset(data_dir=args.data_dir, modality=args.modality,
                                split='train', task=args.task, batch_size=args.batch_size, resize=args.resize)

    val_dataset = get_dataset(data_dir=args.data_dir, modality=args.modality,
                                split='val', task=args.task, batch_size=args.batch_size, resize=args.resize)

    # Early stopping / checkpoint manager
    early_stopping = EarlyCheckpointing(monitor='val_loss', patience=args.patience,
                                        min_delta=args.min_delta,
                                        checkpoint_dir=args.checkpoint_dir,
                                        experiment_name=args.experiment_name,
                                        save_top_k=args.save_top_k,
                                        verbose=True, print_fun=logger.info)

    if args.model_type == "CNN":
        model = CNNModel(input_shape=(args.resize[0], args.resize[1], 1),
                         units=args.units,
                         task=args.task,
                         n_classes=1 if args.task == "binary" else 3,
                         n_conv_layers=args.n_conv_layers,
                         n_dense_layers=args.n_dense_layers,
                         dropout=args.dropout)
    else:
        model = None

    model.summary()

    if args.task == "binary":
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        run_epoch(model, train_dataset, loss_fn, args.task, epoch, args.epochs, optimizer,
                                  training=True)

        if epoch == 1 or epoch % args.check_val_every_n_epochs == 0:
            val_metrics = run_epoch(model, val_dataset, loss_fn, args.task, epoch, training=False)

            if early_stopping(val_metrics["loss"], model, epoch):
                break

    logger.info(f"Epoch {epoch}/{args.epochs} finished training. \n"
                f"Best {args.monitor}: {early_stopping.best_score:.4f} - Best checkpoint saved to {early_stopping.best_ckpt}")


if __name__ == '__main__':
    def parse_tuple(string):
        try:
            items = string.strip("()").split(",")
            return tuple(int(i) for i in items)
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be numbers separated by commas")

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--check_val_every_n_epochs', type=int, default=1, help='Interval between two checkpoint checks')

    parser.add_argument('--project', type=str, default='deep-learning', help='Wandb project name')

    parser.add_argument('--patience', type=int, default=2, help='Patience. Number of epochs without improvement before early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Minimum change in loss to consider as an improvement for early stopping')
    parser.add_argument('--monitor', type=str, default='val_loss', help='Metric to track for early stopping and model checkpointing')

    parser.add_argument('--checkpoint_dir', type=str, default='experiments', help='Directory to save model checkpoints')
    parser.add_argument('--save_top_k', type=int, default=1, help='Maximum number of model checkpoints to save')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment. If None, wandb run name will be used')
    parser.add_argument('--save_weights_only', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Whether to save weights only')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--modality', type=str, choices=['T1', 'T2', 'FLAIR'], default='T1', help='MRI Modality')
    parser.add_argument('--task', type=str, choices=['binary', 'multi-class'], default='binary', help='Task to train the model onto')
    parser.add_argument('--resize', type=parse_tuple, help="Tuple of height and width to which resize the images, e.g. 256,256 or (256,256)")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument("--model_type", type=str, choices=['CNN', 'ViT'], default='CNN')
    parser.add_argument("--units", type=int, default=128, help='Number of hidden units of the first dense layer of the CNN model')
    parser.add_argument("--n_conv_layers", type=int, default=3, help='Number of hidden layers of the CNN model')
    parser.add_argument("--n_dense_layers", type=int, default=2, help='Number of hidden layers of the CNN classifier')
    parser.add_argument("--dropout", type=float, default=0.3, help='Dropout rate')

    args = parser.parse_args()

    main(args)