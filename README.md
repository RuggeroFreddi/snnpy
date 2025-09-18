# 🧠 Spiking Neural Network (SNN) – Reservoir Model

[![PyPI version](https://img.shields.io/pypi/v/snn-reservoir-py.svg)](https://pypi.org/project/snn-reservoir-py/)
[![Python version](https://img.shields.io/pypi/pyversions/snn-reservoir-py.svg)](https://pypi.org/project/snn-reservoir-py/)
[![License](https://img.shields.io/github/license/RuggeroFreddi/snnpy)](./_LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/RuggeroFreddi/snnpy)](https://github.com/RuggeroFreddi/snnpy/commits/main)

> Library available on [PyPI](https://pypi.org/project/snnpy/)  
> ✅ Installable directly with:

```bash 
pip install snn-reservoir-py==2.0.0
```
This class implements a **Spiking Neural Network (SNN)** model based on **LIF neurons** and a **reservoir** architecture, with either a **small-world** topology generated using the **Watts-Strogatz model**, or a **random-uniform** topology generated using an **Erdős–Rényi model**. The typical operational flow is as follows:

1. **Instantiate** an SNN network.
2. **Provide** a binary matrix `input_spike_times` with shape `(num_input_neurons, time_steps)` where `1` indicates stimulation of an input neuron at a specific time.
3. At each timestep, a **current** is injected into the activated input neurons.

> ⚠️ **Input neurons** correspond to the first `input_spike_times.shape[0]` indices.  
> ⚠️ **Output neurons** are randomly selected among hidden neurons (non-input), unless you set them manually via `set_output_neurons()`.

Optionally, the model supports **STDP-based synaptic plasticity**, **synapse pruning**, and an **auto-tuning membrane threshold** mechanism for homeostasis.

STDP is implemented using **pre- and post-synaptic traces** that decay exponentially:

  x_pre(t+1)  =  x_pre(t)  * exp(-Δt / tau_plus)  
  x_post(t+1) =  x_post(t) * exp(-Δt / tau_minus)

When a spike occurs, the corresponding trace is incremented by 1.  
Synaptic weights are updated according to either:

- **Multiplicative rule**  
  LTP: w ← w + η * A_plus * x_pre * (1 - w/W_max)  
  LTD: w ← w * (1 - (η * A_minus / W_max) * x_post)

- **Additive rule**  
  LTP: w ← w + η * A_plus * x_pre  
  LTD: w ← w - η * A_minus * x_post

where tau_plus/tau_minus are decay constants, A_plus/A_minus are learning amplitudes, η is the learning rate, and W_max is the maximum synaptic weight.

## Example

A minimal end-to-end example is available here:  
➡️ **Example repo:** https://github.com/RuggeroFreddi/Example-SNNpy

---

## ⚙️ Configurable Parameters 

### (`SimulationParams`)

| Parameter                | Type               | Description                                                                                     | Restrictions and Defaults                                                                          |
|--------------------------|--------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `num_neurons`            | `int`              | Total number of neurons in the network (input + hidden + output)                                 | ≥ 1. Required only if `adjacency_matrix` is **not** provided                                        |
| `num_output_neurons`     | `int`              | Number of output neurons automatically selected from the reservoir                               | ≥ 1. Required only if `output_neurons` is **not** provided                                          |
| `output_neurons`         | `np.ndarray[int]`  | Indices of manually selected output neurons                                                      | 1D array of integers in `[0, num_neurons)`. Required only if `num_output_neurons` is not provided   |
| `membrane_threshold`     | `float`            | Membrane potential threshold to trigger a spike                                                  | > 0                                                                                                 |
| `leak_coefficient`       | `float`            | Decay factor applied to the membrane potential at each timestep                                  | Range `[0, 1)`                                                                                      |
| `refractory_period`      | `int`              | Number of timesteps after a spike during which a neuron cannot spike again                       | Any real number.                                                                                                 |
| `duration`               | `int`              | Total simulation duration in timesteps                                                           | ≥ 1 if specified. Default: length of `input_spike_times`                                           |
| `input_spike_times`      | `np.ndarray[int]`  | Binary matrix (neurons × time) with external input stimulation                                   | 2D array of 0/1                                                                                     |
| `membrane_potentials`    | `np.ndarray[float]`| Initial membrane potentials                                                                      | 1D array with values in `[0, membrane_threshold]`                                                  |
| `adjacency_matrix`       | `scipy.sparse`     | Weighted adjacency matrix of the neural network                                                  | Square matrix with float weights. Required only if `num_neurons` is not provided                   |
| `mean_weight`            | `float`            | Mean synaptic connection weight                                                                  | > 0. Required only if `adjacency_matrix` is not provided                                            |
| `weight_variance`        | `float`            | Variance of synaptic weights                                                                     | ≥ 0. Required only if `adjacency_matrix` is not provided. Default: `0.1 * mean_weight`              |
| `current_amplitude`      | `float`            | Amplitude of input current                                                                       | Any real number. Default: `membrane_threshold`                                                     |
| `small_world_graph_p`    | `float`            | Rewiring probability for small-world topology (Watts-Strogatz)                                   | Range `[0, 1]`. Required only if `is_random_uniform` is `False`                                    |
| `small_world_graph_k`    | `int`              | Each node initially connected to `k` neighbors in small-world topology                           | Even integer ≥ 2. Required only if `is_random_uniform` is `False`                                  |
| `connection_prob`        | `float`            | Connection probability in random uniform topology                                                | Range `[0, 1]`. Required only if `is_random_uniform` is `True`                                     |
| `is_random_uniform`      | `bool`             | If `True`, generates a random uniform network; otherwise uses small-world topology               | Default: False. Exclusive with `small_world_graph_k` and `small_world_graph_p` if `True`           |
### (`STDPParams`)
| Parameter              | Type      | Description                                                                 | Restrictions and Defaults                                                                 |
|------------------------|-----------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `enabled`              | `bool`    | Enable or disable STDP learning                                             | Default: `True`                                                                          |
| `tau_plus`             | `float`   | Time constant for potentiation (pre → post spike interval)                   | > 0. Default: `0.02`                                                                     |
| `tau_minus`            | `float`   | Time constant for depression (post → pre spike interval)                     | > 0. Default: `0.04`                                                                     |
| `A_plus`               | `float`   | Amplitude of long-term potentiation (LTP)                                    | ≥ 0. Default: `1e-3`                                                                     |
| `A_minus`              | `float`   | Amplitude of long-term depression (LTD)                                      | ≥ 0. Default: `1e-3`, or automatically derived if `lock_A_minus=True`                    |
| `eta`                  | `float`   | Learning rate scaling factor                                                 | ≥ 0. Default: `1.0`                                                                      |
| `W_max`                | `float`   | Maximum synaptic weight                                                     | > 0. Default: `1.0`                                                                      |
| `clip`                 | `bool`    | Clip synaptic weights within `[0, W_max]` after each update                  | Default: `True`                                                                          |
| `nearest_neighbor`     | `bool`    | Restrict updates to the nearest pre/post spike pairs                         | Default: `False`                                                                         |
| `lock_A_minus`         | `bool`    | If `True`, set `A_minus = A_plus * (tau_plus / tau_minus)` automatically. Helps avoiding drift   | Default: `False`                                                                         |
| `weight_update_rule`   | `str`     | Rule for weight updates: `"multiplicative"` or `"additive"`                  | Must be one of `{multiplicative, additive}`. Default: `"multiplicative"`                 |

---

## 🧩 Main Methods

### ▶️ Simulation

- `simulate()`  
  Runs the simulation for the full duration and returns a 2D NumPy array (`np.ndarray`): 
  Returns a binary matrix of shape `[time x output_neurons]`, where each row is a timestep and each column an output neuron (1 = spike, 0 = no spike).

- `get_spike_time_lists_output()`  
  Returns a list of lists: each sublist contains the timesteps at which each output neuron generated a spike.

- `set_input_spike_times(input_spike_times)`  
  Sets the binary input spike matrix (shape `[input_neurons (0/1) x time]`).  
  Automatically updates the simulation duration (`duration`) if not already set.

- `set_membrane_potentials(membrane_potentials)`  
  Sets the initial membrane potentials for each neuron.  
  The array must be 1D with length equal to the number of neurons (`num_neurons`).

- `set_output_neurons(indices)`  
  Sets the output neurons by specifying their indices (array of integers).  

- `reset()`  
  Resets the internal state of the network to allow for a new simulation.  
  Restores initial membrane potentials, clears spike matrix, spike count, and refractory timers, while preserving all other parameters (`adjacency_matrix`, `output_neurons`, etc.).

---

### 🔧 STDP & Homeostasis
These methods implement synaptic plasticity (STDP) and homeostatic regulation (threshold auto-tuning, weight rescaling, pruning).

- `rescale_synaptic_weights_to_mean(target_mean)`  
  Scales all non-zero synaptic weights so that their mean equals the given `target_mean`.  
  Preserves the relative distribution of weights while adapting their global strength.  
  *(Homeostatic mechanism: keeps synaptic activity balanced.)*

- `compute_global_scalar_threshold(use_abs_weights=False)` / `apply_global_scalar_threshold(use_abs_weights=False)`  
  Compute / apply a recommended firing threshold based on:  
  θ = (mean synaptic weight × mean in-degree) + (2 × input_mean_current × refractory_period).  
  *(see [A Mean-Field Approach to Criticality in Spiking Neural Networks for Reservoir Computing](https://www.biorxiv.org/content/10.1101/2025.02.05.636716v1.full.pdf))*  
  *(Homeostatic mechanism: adapts membrane threshold automatically to balance activity.)*

- `disable_stdp()`  
  Disables STDP learning and clears all internal traces and buffers.  
  Useful to run the network in static mode after plasticity has been tested.
  
- `prune(fraction)`  
  Removes the weakest fraction of synapses (by absolute weight).  
  For example, `prune(0.2)` deletes the 20% smallest weights.  
  After pruning, statistics (mean, variance, in-degree) are updated automatically.

- `reset_synaptic_weights(mean, std=0.1)`  
  Resets existing synaptic weights by re-sampling them from a normal distribution with the specified mean and standard deviation (`std * mean`).  Can be combined with STDP to restart learning from a fresh initialization.
  
---

### 📈 Feature Extraction (after simulation)
Each feature extraction method returns a one-dimensional NumPy array (`np.ndarray`) containing one feature per output neuron, except `extract_features_from_spikes()`, which returns a `dict` of feature arrays keyed by feature name (each value is a 1D array over output neurons).


- `extract_features_from_spikes()`
  Extracts all the main features from the output neurons (spike count, entropy, ISI, etc.)

- `get_spike_counts()`  
  Total number of spikes per neuron.

- `get_mean_spike_times()`, `get_first_spike_times()`, `get_last_spike_times()`  
  Temporal statistics on spike timing.

- `get_mean_isi_per_neuron()`, `get_isi_variance_per_neuron()`  
  Statistics on inter-spike intervals (ISI).

- `get_spike_entropy_per_neuron()`  
  Entropy of the temporal distribution of spikes.

- `get_spike_rates()`  
  Average spike rate (spikes per time unit).

- `get_autocorrelation_first_lag()`  
  Autocorrelation at lag 1 (indicator of regularity).

- `get_burstiness()`  
  Burstiness index (ISI variability).

- `get_spike_symmetry()`  
  Spike symmetry between the first and second half of the timeline.

- `get_spike_histogram_moments()`  
  Mean, skewness, and kurtosis of the spike time histogram.

- `get_burst_counts()`  
  Number of bursts per neuron. A burst is defined as a contiguous sequence of spikes (value 1) in the neuron's spike train. Useful for assessing the frequency of temporally clustered activity.

---

### 🧪 Utility

- `calculate_mean_isi()`  
  Returns the average ISI across all neurons.

- `reset_synaptic_weights(mean, std)`  
  Regenerates all existing synaptic weights from a normal distribution.

---

### 💾 Saving/Loading

- `save_topology()`, `load_topology()`  
  Save/load the synaptic weight matrix (`adjacency_matrix`) to/from disk in `.npz` format (SciPy sparse matrix).

- `save_membrane_potentials()`, `load_membrane_potentials()`  
  Save/load membrane potentials in `.npy` format.

- `save_output_neurons()`, `load_output_neurons()`  
  Save/load the indices of the output neurons in `.npy` format.

- `set_topology(sparse_matrix)`, `get_topology()`  
  Set or retrieve the synaptic weight matrix. 

- `set_membrane_potentials(array)`, `get_membrane_potentials()`  
  Set or retrieve the membrane potentials.

- `set_output_neurons(indices)`, `get_output_neurons()`  
  Set or retrieve the indices of the output neurons.

> ℹ️ **Note:** The methods `load_topology()`, `load_output_neurons()`, and `load_membrane_potentials()`  
> are also available as **standalone functions**, useful for loading data **before** initializing the `SNN` object.

---

## 📁 Output Files

| Function                  | Default Path                           |
|--------------------------|----------------------------------------|
| Synaptic topology         | `dati/snn_matrices.npz`                |
| Membrane potentials       | `dati/membrane_potentials.npy`         |
| Output neurons            | `dati/output_neurons.npy`              |

---

## ▶️ Example Usage

```python
from snnpy import SNN, SimulationParams
import numpy as np

# Input spike train (input neurons x duration)
my_input = np.random.randint(0, 2, size=(50, 500), dtype=np.uint8)

# Parameter configuration
params = SimulationParams(
    num_neurons=2000,
    num_output_neurons=35,
    input_spike_times=my_input,
    leak_coefficient=1 / 10000,
    refractory_period=2,
    membrane_threshold=2.0,
    is_random_uniform=False,
    small_world_graph_p=0.2,
    small_world_graph_k=int(0.10 * 2000 * 2),
    mean_weight=0.00745167232 * 1.05
)

# Create and simulate the network
snn = SNN(params)
output = snn.simulate()

# Extract temporal features from output neurons
features = snn.extract_features_from_spikes()

print("Output shape:", output.shape)

print("Available feature keys:", list(features.keys()))
print("Spike counts shape:", features["spike_counts"].shape)


