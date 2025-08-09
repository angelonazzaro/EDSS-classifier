import glob
import logging
import os
from typing import Callable, Optional, Literal

logging.basicConfig(level=logging.INFO)


class EarlyCheckpointing:
    """
        Early stopping + Model checkpointing callbacks.
        Monitors model improvement over a metric during training, checkpointing the best versions.
        Halts training if the tracked metric does improve over a specified period.
    """
    def __init__(self, monitor, obj: Literal['minimize', 'maximize'] = "minimize",
                 min_delta: float = 0.01, patience: int = 5, verbose: bool = False,
                 checkpoint_dir: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 save_top_k: int = 1,
                 save_weights_only: bool = False,
                 print_fun: Optional[Callable] = None):

        self.monitor = monitor
        self.obj = obj
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.print_fun = print_fun

        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only

        os.makedirs(checkpoint_dir, exist_ok=True)

        if self.verbose and self.print_fun is None:
            logging.warning("Early stopping is in verbose mode but not `print_fun` is specified. Defaulting to `print` to stdout.")
            self.print_fun = print
        elif not self.verbose:
            logging.warning("Early stopping is not in verbose mode. `print_fun` is ignored.")

        self.counter = 0
        self.best_score = float('inf') if self.obj == 'minimize' else float('-inf')
        self.best_ckpt = None

    def __call__(self, val, model, epoch):
        if self.obj == 'minimize':
            improvement = self.best_score - val
        else:
            improvement = val - self.best_score

        if improvement >= self.min_delta:
            if self.verbose:
                self.print_fun(f"{self.monitor} improved from {self.best_score} to {val}.")
            self.best_score = val
            self.counter = 0
            self._save_checkpoint(val, model, epoch)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.print_fun(
                    f"{self.monitor} did not improve for {self.patience} epochs. Stopping training...."
                )
                return True
            if self.verbose:
                self.print_fun(
                    f"{self.monitor} did not improve. "
                    f"Early stopping counter: {self.counter} out of {self.patience}"
                )

        return False

    def _save_checkpoint(self, val, model, epoch):
        ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}_score_{val:.4f}.ckpt")

        if self.save_weights_only:
            model.save_weights(ckpt_path)
        else:
            model.save(ckpt_path) # save hyperparameters, architecture and optimizers state

        if self.verbose:
            self.print_fun(f"Saved checkpoint to: {ckpt_path}")

        self.best_ckpt = ckpt_path

        # Collect and sort existing checkpoints
        current_checkpoints = glob.glob(f"{self.checkpoint_dir}/*.ckpt")
        checkpoints_with_scores = []

        for ckpt in current_checkpoints:
            try:
                score_str = ckpt.split("_score_")[-1].replace(".ckpt", "")
                ckpt_score = float(score_str)
                ckpt_score = ckpt_score if self.obj == "maximize" else -ckpt_score
                checkpoints_with_scores.append((ckpt_score, ckpt))
            except ValueError:
                continue  # Ignore files not matching naming pattern

        # Sort by score (descending)
        checkpoints_with_scores.sort(reverse=True, key=lambda x: x[0])

        # If too many checkpoints, remove the worst ones
        if len(checkpoints_with_scores) > self.save_top_k:
            for _, path_to_remove in checkpoints_with_scores[self.save_top_k:]:
                os.remove(path_to_remove)
                if self.verbose:
                    self.print_fun(f"Removed old checkpoint: {path_to_remove}")
