# Introduction
Multiple sclerosis (MS) is a chronic, often disabling disease that affects the central nervous system. It is characterized
by the development of scar tissue (sclerosis) in place of the normal
tissue component of the nervous system, interfering with the transmission of nerve impulses.

The **Expanded Disability Scale** (**EDSS**) is a scale designed to assess the
levels of disability of people with MS; it ranges from 0, corresponding to a normal
neurological exam, to 10. It includes intermediate and increasingly
higher levels of disability. The score is obtained by adding the partial scores of the
various functional systems related to nervous system activity (pyramidal,
cerebellar, sphincter, etc.). This allows for easier assessment
of the progression of the disease and allows for verification of the effectiveness of the
current therapy.

In the remainder of this work, we will analyze the classification of patients' disability levels in two contexts:
- **Binary classification**: Patients are grouped into two classes based on their EDSS score, namely the _positive_ class, which includes all patients with an EDSS value less than or equal to 2.0, and the _negative_ class, with all patients with an EDSS value greater than 2.0. A patient belonging to the positive class is one who has no significant lesions and minimal or no functional impairment, while the _negative_ class includes patients with obvious neurological lesions and clinical symptoms.

- **Multiclass classification**: EDSS scores were mapped into three categories: _normal_, _mild_, and _severe_. Specifically, scores from 0 to 2.0 were labeled as normal, indicating mild to no disability; Scores between 2.5 and 4.0 were labeled as mild, indicating patients with moderate impairment but preserved ambulation, while scores above 4.0 were labeled as severe, corresponding to individuals with significant motor or systemic dysfunction.

To address this task, we will develop and compare two types of
models: one based on Convolutional Neural Networks (CNN) and one on Vision
Transformer (ViT)}. Since we have
a limited dataset and the intended use is clinical, we will focus on
"lightweight" models that can be deployed in environments with
limited computational resources. Furthermore, the use of complex models
may not be suitable in an application context such as the medical one, where
data is scarce due to high acquisition and annotation costs and ongoing
patient privacy issues and administrative policies that impact
data sharing.

# Dataset

The dataset employed in this work consists of T1, T2 and FLAIR cerebral MRI sequences. The dataset with pre-defined training, validation and test splits for each task is available under: `edss-dataset.zip`.

For more information, refer to the documentation under the `deliverables` folder.

# Models' Weights

The models' weights are available under `weights.zip`. Note that only the best models were uploaded.

For more information, refer to the documentation under the `deliverables` folder.

# Installation Guide

## Installing Python

Verify you have Python installed on your machine. This project is compatible with Python `3.11`. 
Run `python --version` to check what python version you have installed.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).

## Creating the Virtual Environment

It's strongly recommended to create a virtual environment before proceeding with the installation. 
We recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) for major information.

## Cloning the Repository

Clone the repository by running the following command in your terminal:

```bash
git clone https://github.com/angelonazzaro/EDSS-classifier.git
```

## Installing Requirements

Before installating the requirements, **make sure you have your virtual environment activated**. If that's the case, you terminal should look something like: `(name-of-your-venv) user@userpath`.

Install the requirements using `pip`:

```bash
pip install -r requirements.txt
```

**NOTE**: If you have an Nvidia GPU, overwrite `tensorflow==2.19.0` with `'tensorflow[and-cuda]==2.19.0'`. If you have an Apple Silicon Chip (M1, M2, M3 or M4), please _also_ install `tensorflow-metal` for MPS support.

**NOTE**: A [Weight and Biases](https://wandb.ai/) account is needed for experiment tracking.

# Training 

##  Usage

To start training you need to execute the `train.py` script. The script is executed from the command line:

```bash
python train.py --data_dir <path_to_data> [OPTIONS]
```

### Required Argument
- `--data_dir`: Path to the dataset directory.
- `--resize`: Size to resize the images to. A tuple of ints.
---

## ⚙️ Command-line Arguments

### General Settings
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--epochs` | int | 10 | Number of training epochs |
| `--check_val_every_n_epochs` | int | 1 | Interval for running validation |
| `--batch_size` | int | 32 | Batch size |
| `--resize` | tuple | None | Resize dimensions `(height, width)` |
| `--include_augmented` | bool | False | Include augmented images |
| `--model_type` | str | CNN | Model type (`CNN` or `ViT`) |

---

### Model Settings
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **CNN-specific** |
| `--units` | int | 128 | Hidden units in the first dense layer |
| `--n_conv_layers` | int | 3 | Number of convolutional layers |
| `--n_dense_layers` | int | 2 | Number of dense layers in classifier |
| **ViT-specific** |
| `--patch_size` | int | None | Patch size |
| `--d_model` | int | None | Embedding dimension |
| `--mlp_dim` | int | None | MLP hidden units |
| `--num_heads` | int | 8 | Number of attention heads |
| `--n_layers` | int | 6 | Number of encoder layers |
| `--channels` | int | 1 | Number of channels |

---

### Task & Loss Settings
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | binary | Task type (`binary` or `multi-class`) |
| `--alpha` | float | 0.25 | Alpha for Focal Loss |
| `--gamma` | float | 2.0 | Gamma for Focal Loss |

---

### Optimization
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | 1e-3 | Learning rate |
| `--weight_decay` | float | 1e-5 | Weight decay for AdamW |

---

### Early Stopping & Checkpoints
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--patience` | int | 2 | Early stopping patience |
| `--min_delta` | float | 0.0001 | Minimum improvement in monitored metric |
| `--monitor` | str | val_loss | Metric to monitor |
| `--checkpoint_dir` | str | experiments | Directory to save checkpoints |
| `--save_top_k` | int | 1 | Max checkpoints to keep |
| `--experiment_name` | str | None | Experiment name (W&B run name if None) |

---

### Weights & Biases (wandb)
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--project` | str | deep-learning | W&B project name |
| `--tune_hyperparameters` | bool | False | Enable W&B hyperparameter sweep |
| `--sweep_config` | str | sweep.yaml | Path to sweep configuration file |
| `--sweep_count` | int | 10 | Number of sweep runs |

---

### Example Commands

#### Basic CNN Training
```bash
python train.py   --data_dir ./data   --model_type CNN   --resize 256,256   --epochs 20   --batch_size 64
```

#### Vision Transformer Training
```bash
python train.py   --data_dir ./data   --model_type ViT   --patch_size 16   --d_model 128   --mlp_dim 256   --epochs 30
```

#### Binary Classification with Augmented Data
```bash
python train.py   --data_dir ./data   --task binary   --include_augmented   --resize 224,224
```

#### Run Hyperparameter Sweep with W&B
```bash
python train.py   --data_dir ./data   --tune_hyperparameters   --sweep_config sweep.yaml   --sweep_count 20
```

---

##  Notes
- If `--experiment_name` is not provided, the W&B run name will be used for saving checkpoints.
- For multi-class classification, `CategoricalFocalCrossentropy` is used; for binary classification, `BinaryCrossentropy` is used.


# Testing 

##  Usage

To start training you need to execute the `test.py` script. The script is executed from the command line:

```bash
python test.py --data_dir <path_to_data> [OPTIONS]
```


### General Settings
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--batch_size` | int | 32 | Batch size |
| `--resize` | tuple | None | Resize dimensions `(height, width)` |
| `--model_type` | str | CNN | Model type (`CNN` or `ViT`) |
---

### Data & Task Settings
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | **Required** | Path to dataset directory |
| `--modality` | str list | T1 | MRI modality (`T1`, `T2`, `FLAIR`) |
| `--task` | str | binary | Task type (`binary` or `multi-class`) |

---

### Model Settings
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | **Required** | Model name |
| `--checkpoint_path` | str | **Required** | Path to model checkpoint |

---

### Results Saving
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--results_dir` | str | results | Directory to save results and confusion matrices |

---

## Output

When executed, the script will:
1. Load the model checkpoint.
2. Run inference on the test dataset.
3. Compute **accuracy**, **precision**, **recall**, and **F1 score**.
4. Save results in a CSV file:  
   `results/<task>_results.csv`
5. Save confusion matrix plots in:  
   `results/<task>/<model_name>/cm_<modality>.png`

---

## Example Commands

#### Test a CNN Model (Binary Classification)
```bash
python test.py   --data_dir ./data   --model_name cnn_model_v1   --checkpoint_path ./experiments/cnn_model_v1/best_model.h5   --task binary   --modality T1   --resize 256,256
```

#### Test a ViT Model (Multi-class Classification)
```bash
python test.py   --data_dir ./data   --model_name vit_model_v1   --checkpoint_path ./experiments/vit_model_v1/best_model.h5   --task multi-class   --modality FLAIR   --resize 224,224
```

