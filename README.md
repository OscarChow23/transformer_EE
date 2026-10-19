# transformer_EE
A transformer encoder based neutrino energy estimator. This is a highly flexible framework which allows for easy modification of the model, loss function, prediction targets, data loader, etc. Currently, the code supports CSV file inputs with arrays stored as a comma separated string. We provide example scripts and configuration files for initial use within NOvA and DUNE experiments.

## Features

* Transformer encoder based model
* Customizable loss function (you can in principle design quite complex loss functions)
* Customizable model architecture
* Customizable optimizer
* Customizable input features and output targets (w/in the highly customizable loss function)

## Prerequisites

Though not always necessary (for instance, if running within the Fermi National Accelerator Laboratory Elastic Analysis Facility), it is recommended to utilize a Python virtual environment/container to run the code training or inference time.

* Python 3.10 was used for development (higher versions should be compatible).
* PyTorch 2.4.1 was used for development (higher versions should be compatible). Pytorch 2.4.1 supports both CUDA and Apple MPS. By default, the code will use CUDA if it is available. Otherwise, it will use Apple MPS or CPU.
* Pandas (default dataframe library)
* [Polars](https://pola.rs/) (an optional faster dataframe library). We recommend Polars for better memory efficiency and faster data preprocessing.
* Numpy
* Matplotlib >= 3.5.

### Example conda setup in a Linux system with NVIDIA GPU
Conda is a Python environment manager which allows user to create an independent virtual environment and install Python packages.
1. make sure you have conda installed in your computer. See https://docs.anaconda.com/free/miniconda/miniconda-install/ for more details.
2. create a new conda environment by running `conda create --name new_env_name`
3. activate the new environment by running `conda activate new_env_name`
4. install the required packages by running `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`, check https://pytorch.org/get-started/locally/ for the latest stable versions of PyTorch and CUDA.
5. install other required packages by running `conda install scipy pandas polars numpy matplotlib`

## (optional) Set up PYTHONPATH
User can directly run training script without setting up environment variables by putting their script in the transformer_EE directory. The following steps are optional.

PYTHONPATH should be set to the top level directory of the repository.

For example, if this repo is cloned to `/home/user/transformer_EE`, then the following should be added to your `.bashrc` file if you are using bash in Linux:
```
export PYTHONPATH=/home/user/transformer_EE:$PYTHONPATH
```

Test that the `PYTHONPATH` is set correctly by running the following command:
```
echo $PYTHONPATH
```
The commandline should return:
```
/home/user/transformer_EE:
```

## Running the code
To run the code and start training, you can run the following command:
```
python3 train_script.py
```

For the flexible CLI launcher used in many training campaigns:
```
python3 train_wide.py --base-config <config.json> --epochs 50
```

### Training input noise

The optional top-level `noise` block supports `mode: "coherent"` (one fractional
shift per event/slice, which maintains the past baseline nominal behavior of the
Transformer_EE) alongside the newer `mode: "particle"` (an independent
fractional shift per particle within each event). In this case, each particle's
draw is shared across its noised training feature components. Both modes support
Gaussian and uniform noise distributions. Only training batches are augmented,
while of course targets are unchanged. Of course, n

oise is not used during inference.

For a config that already selects noisy features, switch to independent particle
shifts and leave scalars unchanged with:

```bash
python3 train_wide.py --base-config <config.json> \
  --noise-mode particle --noise-scalar-mode none
```

Add `--noise-name uniform --noise-low -0.2 --noise-high 0.2` for uniform fractional
shifts between -20% and +20%. Gaussian `--noise-std 0.2` instead means a standard
deviation of 20%; it is not a bound. Event-level scalars can share the coherent
vector draw (`--noise-scalar-mode coherent`, requiring coherent vector mode),
receive an independent draw for each selected scalar (`independent`), or share
the arithmetic mean of the actual particle draws (`particle_mean`, requiring
particle vector mode). The default `auto` shares the coherent event draw in
coherent mode and leaves scalars unchanged in particle mode. The particle mean
excludes padding and is not an energy-weighted recomputation of a scalar total.

Explicit CLI noise options at runtime will override any corresponding JSON values,
while omitted options preserve them. Use `--enable-noise` to add a block or
`--disable-noise` to remove it for a run.

See [Training input noise](docs/noise.md) for complete JSON examples, CLI options,
scalar handling, and the distinction between independent particles and a
species-dependent calibration variation.

### Checkpoint behavior

Training writes checkpoint files with clear use cases:

* `best_model.zip`: lowest validation-loss model observed so far
* `last_model.zip`: latest epoch model (removed automatically if it is identical to the best epoch)
* Optional rolling epoch snapshots: `last_model_epoch_<i>.zip` (configurable retention)
* `checkpoint_metadata.json`: a compact metadata file with best/last epoch and validation loss,
  retained rolling checkpoint names, and early-stopping status.

The best-loss sentinel is initialized scale-independently using `+inf` (instead of a fixed
numeric threshold), which is robust for large-magnitude losses such as MACE-based objectives.

### Early stopping (train_wide.py)

`train_wide.py` now supports optional early stopping based on percentage validation-loss
improvements over a configurable trailing window:

* `--early-stop-window N`: enable early stopping with a trailing window of `N` epochs (for intsance: `N >= 2`)
* `--early-stop-min-delta-pct P`: minimum required percentage improvement over the window (must be > 0 when early stopping is enabled; defaults to `0.1`)
* `--keep-last-n-epoch-checkpoints K`: keep a rolling set of `K` epoch-specific `last_model_epoch_<i>.zip` checkpoints

If early stopping is not enabled, training behavior is unchanged apart from richer
checkpointing metadata and safer evaluation fallback behavior than previous iterations.

## Configuring the code
The config file is a json file, with the default locations placed at transformer_ee/config
There are two ways to configure the code:
1. Edit the config file directly (not recommended as these files could be used by other scripts)
2. Modify the dictionary in the train_script.py file. For example, to select the model, add the following line:
```
input_d["model"]["name"] = "Transformer_EE_MV"
```
in the train_script.py file.

## Logging

Transformer_EE supports WandB logging. To enable WandB logging, users should install WandB
by running `pip install wandb` followed by `wandb init` and type in their API key.

A minimal example of using WandB logging is provided in the train_script.py file.

## More

The above information should help users set up the model training environment and run the script.
If you want to know more details about (past) code structure developments, please check out the [Wiki page](https://github.com/wswxyq/transformer_EE/wiki).
