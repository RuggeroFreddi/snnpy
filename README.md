# 🧠 Spiking Neural Network (SNN) – Reservoir Model

> Library available on [PyPI](https://pypi.org/project/snnpy/)  
> ✅ Installable directly with:

```bash 
pip install snnpy
```
This class implements a **Spiking Neural Network (SNN)** model based on **LIF neurons** and a **reservoir** architecture, with either a **small-world** topology generated using the **Watts-Strogatz model**, or a **random-uniform** topology generated using an **Erdős–Rényi model**. The typical operational flow is as follows:

1. **Instantiate** an SNN network.
2. **Provide** a binary matrix `input_spike_times` with shape `(num_input_neurons, time_steps)` where `1` indicates stimulation of an input neuron at a specific time.
3. At each timestep, a **current** is injected into the activated input neurons.

> ⚠️ **Input neurons** are randomly selected, with their count equal to the number of rows in `input_spike_times`.  
> ⚠️ **Output neurons** are randomly selected from the hidden neurons (non input neurons), but can also be manually specified by passing an index array to `set_output_neurons()`.

---

## ⚙️ Configurable Parameters (`SimulationParams`)

| Parameter                | Type               | Description                                                                                     | Restrictions and Defaults                                                                          |
|--------------------------|--------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `num_neurons`            | `int`              | Total number of neurons in the network (input + hidden + output)                                 | ≥ 1. Required only if `adjacency_matrix` is **not** provided                                        |
| `num_output_neurons`     | `int`              | Number of output neurons automatically selected from the reservoir                               | ≥ 1. Required only if `output_neurons` is **not** provided                                          |
| `output_neurons`         | `np.ndarray[int]`  | Indices of manually selected output neurons                                                      | 1D array of integers in `[0, num_neurons)`. Required only if `num_output_neurons` is not provided   |
| `membrane_threshold`     | `float`            | Membrane potential threshold to trigger a spike                                                  | > 0                                                                                                 |
| `leak_coefficient`       | `float`            | Decay factor applied to the membrane potential at each timestep                                  | Range `[0, 1)`                                                                                      |
| `refractory_period`      | `int`              | Number of timesteps after a spike during which a neuron cannot spike again                       | ≥ 0                                                                                                 |
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

---

## 🧩 Main Methods

### ▶️ Simulation

- `simulate()`  
  Runs the simulation for the full duration and returns a 2D NumPy array (`np.ndarray`):
  
  - A binary matrix of shape `[output_neurons x time]`, where each row represents a timestep and each column an output neuron. The entry is `1` if the neuron spiked at that time, `0` otherwise.

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

### 📈 Feature Extraction (after simulation)
Each feature extraction method returns a one-dimensional NumPy array (`np.ndarray`) containing one feature per output neuron, except `extract_features_from_spikes()`, which returns a two-dimensional array with multiple features per neuron.

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
  Set or retrieve the synaptic weight matrix. *(Note: `set_topology()` is not explicitly implemented but can be easily added.)*

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
print("Feature vector shape:", features.shape)

