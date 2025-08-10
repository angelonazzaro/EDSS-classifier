import argparse
import logging
import os
from functools import partial

import tensorflow as tf
import yaml
from tqdm import tqdm

import wandb
from edss_dataset import get_dataset
from model.cnn import CNNModel
from model.vit import VisionTransformer
from utils import parse_tuple
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


def train(args):
    run = wandb.init(project=args.project, config=dict(vars(args)), name=args.experiment_name)

    if args.experiment_name is None:
        args.experiment_name = wandb.run.name

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(os.path.join(args.checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(dict(vars(args)), f, default_flow_style=False, sort_keys=False)

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
        model = VisionTransformer(image_size=args.resize[0], patch_size=args.patch_size, d_model=args.d_model,
                                  mlp_dim=args.mlp_dim, num_heads=args.num_heads,
                                  dropout=args.dropout, n_layers=args.n_layers, task=args.task,
                                  n_classes=1 if args.task == "binary" else 3)

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

    run.finish()


def main(args):
    # ensure reproducible results
    tf.keras.utils.set_random_seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    args.sweep_id = None
    if args.tune_hyperparameters:
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(
            sweep=sweep_config, project=args.project)
        sweep_config.update(vars(args))
        args.sweep_id = sweep_id
        wandb.agent(sweep_id, partial(train, args), count=args.sweep_count)
    else:
        train(args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Seed for random generation')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--check_val_every_n_epochs', type=int, default=1,
                        help='Interval between two checkpoint checks')

    parser.add_argument('--project', type=str, default='deep-learning', help='Wandb project name')
    parser.add_argument('--tune_hyperparameters', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to tune hyperparameters')
    parser.add_argument("--sweep_config", type=str, default='sweep.yaml', help='YAML file containing the wandb sweep '
                                                                               'configuration for hyper-parameter '
                                                                               'search')
    parser.add_argument("--sweep_count", type=int, default=10, help='Number of tries for the sweep')

    parser.add_argument('--patience', type=int, default=2,
                        help='Patience. Number of epochs without improvement before early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0001,
                        help='Minimum change in loss to consider as an improvement for early stopping')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        help='Metric to track for early stopping and model checkpointing')

    parser.add_argument('--checkpoint_dir', type=str, default='experiments', help='Directory to save model checkpoints')
    parser.add_argument('--save_top_k', type=int, default=1, help='Maximum number of model checkpoints to save')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment. If None, wandb run name will be used')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

    parser.add_argument('--data_dir', required=True, type=str, help='Data directory')
    parser.add_argument('--modality', type=str, choices=['T1', 'T2', 'FLAIR'], nargs='+', default='T1',
                        help='MRI Modality')
    parser.add_argument('--task', type=str, choices=['binary', 'multi-class'], default='binary',
                        help='Task to train the model onto')
    parser.add_argument('--resize', type=parse_tuple,
                        help="Tuple of height and width to which resize the images, e.g. 256,256 or (256,256)")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument("--model_type", type=str, choices=['CNN', 'ViT'], default='CNN')

    parser.add_argument("--units", type=int, default=128,
                        help='Number of hidden units of the first dense layer of the CNN model')
    parser.add_argument("--n_conv_layers", type=int, default=3, help='Number of hidden layers of the CNN model')
    parser.add_argument("--n_dense_layers", type=int, default=2, help='Number of hidden layers of the CNN classifier')
    parser.add_argument("--dropout", type=float, default=0.3, help='Dropout rate')

    parser.add_argument("--patch_size", type=int, default=None, help="Patch dimension for ViT")
    parser.add_argument("--d_model", type=int, default=None, help="Hidden dimensionality of ViT")
    parser.add_argument("--mlp_dim", type=int, default=None, help="MLP units of ViT")
    parser.add_argument("--num_heads", type=int, default=8, help="Num of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Num of encoder layers for ViT")
    parser.add_argument("--channels", type=int, default=1, help="Num of channels for ViT")

    args = parser.parse_args()

    main(args)