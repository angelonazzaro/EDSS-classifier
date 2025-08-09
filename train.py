import argparseepochs
import logging

import wandb
from tqdm import tqdm

from edss_dataset import get_dataset
from utils.early_monitoring import EarlyCheckpointing

logger = logging.getLogger(__name__)


def val(model, dataset, criterion):
    for step, (x_batch, y_batch) in enumerate(tqdm(dataset, desc=f"Validation Step")):
        pass

def train(model, dataset, optimizer, criterion, epoch, epochs):
    for step, (x_batch, y_batch) in enumerate(tqdm(dataset, desc=f"Training Epoch {epoch}/{epochs}")):
        pass

def main(args):
    wandb.init(entity=args.project, project=args.project, config=dict(args), name=args.experiment_name)

    if args.experiment_name is None:
        args.experiment_name = wandb.run.name

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
    # TODO: define model
    model = None
    optimizer = None
    loss_fn = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_dataset, optimizer, loss_fn, epoch, args.epochs + 1)

        if epoch == 1 or epoch % args.check_val_every_n_epochs == 0:
            val_loss = val(model, val_dataset, loss_fn)

            if early_stopping(val_loss, model, epoch):
                break

    logger.info(f"Epoch {epoch}/{args.epochs} finished training. \n"
                f"Best {args.monitor}: {early_stopping.best_score:.4f} - Best checkpoint saved to {early_stopping.best_ckpt}")


if __name__ == '__main__':
    def parse_tuple(string):
        try:
            items = string.strip("()").split(",")
            return tuple(float(i) for i in items)
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be numbers separated by commas")

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--check_val_every_n_epochs', type=int, default=1, help='Interval between two checkpoint checks')

    parser.add_argument('--project', type=str, default='deep-learning', help='Wandb project name')

    parser.add_argument('--patience', type=int, default=2, help='Patience. Number of epochs without improvement before early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Minimum change in loss to consider as an improvement for early stopping')
    parser.add_argument('--monitor', type=str, default='val_loss', help='Metric to track for early stopping and model checkpointing')

    parser.add_argument('--checkpoint_dir', type=str, default='model', help='Directory to save model checkpoints')
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

    args = parser.parse_args()

    main(args)