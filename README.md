# TwinProp

This repository contains the code accompanying the preprint:

**["title"](https://link)**

TwinProp provides code for simulating a biophysically detailed neuron model, training a deep neural network (DNN) twin for it, and utilizing it for computations via the TwinProp algorithm.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Compiling the Neuron Model](#compiling-the-neuron-model)
3. [Step 1: Create a Simulation Dataset](#step-1-create-a-simulation-dataset)
4. [Step 2: Train a Neural Network Twin](#step-2-train-a-neural-network-twin)
5. [Step 3: Utilize the Neuron for a Task](#step-3-utilize-the-neuron-for-a-task)
6. [Step 4: Verify the Solution](#step-4-verify-the-solution)

---

## Prerequisites

- Python 3.8+
- NEURON simulator installed (for biophysical simulations)
- PyTorch
- NumPy, SciPy, h5py, matplotlib, scikit-learn
- PyTorch Lightning
- wandb (optional, for logging)

---

## Compiling the Neuron Model

Before running simulations, you must compile the NEURON model's MOD files. Navigate to the neuron model folder and run `nrnivmodl` on the `mods` directory:

```bash
cd simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay
nrnivmodl mods
```

This compiles the mechanism files and creates an architecture-specific folder (e.g., `x86_64/`) containing the compiled files. You must do this once for each neuron model before running simulations.

---

## Step 1: Create a Simulation Dataset

Generate a dataset of neuron simulations using `submit_simulate_neuron_and_create_dataset.py`. This script submits simulation jobs and creates train/valid/test datasets.

### Example Command

```bash
python simulating_neurons/submit_simulate_neuron_and_create_dataset.py \
    --neuron_model_folder simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay \
    --simulation_dataset_folder test_sim_hay \
    --simulation_dataset_name hay_dataset \
    --count_simulations_for_train 20 \
    --count_simulations_for_valid 10 \
    --count_simulations_for_test 10 \
    --simulation_duration_in_seconds 10
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--neuron_model_folder` | Path to the neuron model folder | Required |
| `--simulation_dataset_folder` | Output folder for the dataset | Required |
| `--simulation_dataset_name` | Name of the dataset | Optional |
| `--count_simulations_for_train` | Number of training simulations | 20 |
| `--count_simulations_for_valid` | Number of validation simulations | 10 |
| `--count_simulations_for_test` | Number of test simulations | 10 |
| `--simulation_duration_in_seconds` | Duration of each simulation | - |

---

## Step 2: Train a Neural Network Twin

Train a DNN twin on the simulation dataset using `train_neuron_nn.py`. The network learns to predict the neuron's output spikes and somatic voltage.

### Example Command

```bash
python training_nets/train_neuron_nn.py \
    --simulation_dataset_folder test_sim_hay \
    --neuron_nn_folder test_net_hay \
    --neuron_nn_name hay_nn \
    --batch_size 128 \
    --lr 0.007 \
    --maximum_train_steps 10000000
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--simulation_dataset_folder` | Path to the simulation dataset | Required |
| `--neuron_nn_folder` | Output folder for the trained model | Required |
| `--neuron_nn_name` | Name for the neural network | Optional |
| `--batch_size` | Training batch size | 128 |
| `--lr` | Learning rate | 0.007 |
| `--use_elm` | Use Expressive Leaky Memory model | True |
| `--maximum_train_steps` | Maximum training steps | 10000000 |
| `--run_on_gpu` | Enable GPU training | True |
| `--v_loss_weight` | Weight for voltage loss | 0.02 |

The trained models will be saved in `<neuron_nn_folder>/models/`.

---

## Step 3: Utilize the Neuron for a Task

Use the trained DNN twin to solve a task (e.g., image classification) via the TwinProp algorithm using `utilize_neuron.py`.

### Example Command

```bash
python utilizing_neurons/utilize_neuron.py \
    --utilize_neuron_folder test_ut_hay \
    --neuron_model_folder simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay \
    --neuron_nn_file test_net_hay/models/model_9_3230 \
    --spiking_cat_and_dog \
    --count_exc_axons 6400 \
    --count_inh_axons 6400 \
    --count_exc_bias_axons 0 \
    --count_inh_bias_axons 0 \
    --stimulus_duration_in_ms 256 \
    --presampled False \
    --max_count_samples 100 \
    --use_wiring_layer True \
    --count_epochs 10
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--utilize_neuron_folder` | Output folder for results | Required |
| `--neuron_nn_file` | Path to the trained NN model file | Required |
| `--neuron_model_folder` | Path to the neuron model (for verification) | Optional |
| `--spiking_cat_and_dog` | Use spiking cats vs dogs dataset | False |
| `--count_exc_axons` | Number of excitatory axons | 800 |
| `--count_inh_axons` | Number of inhibitory axons | 200 |
| `--stimulus_duration_in_ms` | Stimulus duration in milliseconds | 420 |
| `--use_wiring_layer` | Enable the wiring layer | False |
| `--count_epochs` | Number of training epochs | 10 |
| `--batch_size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |

### Available Datasets

- `--spiking_cat_and_dog`: Spiking cats vs dogs image classification
- `--spiking_afhq`: Spiking AFHQ dataset
- `--shd`: Spiking Heidelberg Digits (audio)
- `--ssc`: Spiking Speech Commands
- `--spiking_abstract`: Abstract pattern classification

---

## Step 4: Verify the Solution

Verify the learned solution by running it on the actual biophysical neuron model. This uses the same `utilize_neuron.py` script but with the biophysical model instead of the DNN twin.

### Example Command

```bash
python utilizing_neurons/utilize_neuron.py \
    --utilize_neuron_folder test_ut_hay_verify \
    --neuron_model_folder simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay \
    --utilizer_from_checkpoint test_ut_hay/logs/lightning_logs/version_0/checkpoints/last.ckpt \
    --spiking_cat_and_dog \
    --count_exc_axons 6400 \
    --count_inh_axons 6400 \
    --count_exc_bias_axons 0 \
    --count_inh_bias_axons 0 \
    --stimulus_duration_in_ms 256 \
    --presampled False \
    --max_count_samples 100 \
    --only_calculate_metrics True
```

### Key Arguments for Verification

| Argument | Description |
|----------|-------------|
| `--neuron_model_folder` | Path to the biophysical neuron model (uses actual simulation) |
| `--utilizer_from_checkpoint` | Path to the checkpoint from Step 3 |
| `--only_calculate_metrics` | Only calculate metrics without further training |

This runs the learned wiring solution on the biophysical neuron model and reports the accuracy, AUC, and MAE metrics to verify that the DNN twin's solution transfers to the real neuron.
