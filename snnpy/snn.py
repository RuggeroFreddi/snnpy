"""Spiking Neural Network (SNN) model."""

import os
import warnings
from functools import wraps
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import networkx as nx
from scipy.sparse import issparse, save_npz, load_npz, csr_matrix, spmatrix



DEFAULT_MATRIX_PATH = "dati/snn_matrices.npz"
DEFAULT_POTENTIALS_PATH = "dati/membrane_potentials.npy"
DEFAULT_OUTPUT_NEURONS_PATH = "dati/output_neurons.npy"


@dataclass
class SimulationParams:
    """Parameters for running the Spiking Neural Network (SNN) simulation."""

    membrane_threshold: float
    leak_coefficient: float
    refractory_period: int
    small_world_graph_p: Optional[float] = None
    small_world_graph_k: Optional[float] = None
    connection_prob: Optional[float] = None
    is_random_uniform: Optional[bool] = None
    mean_weight: Optional[float] = None
    num_neurons: Optional[int] = None
    num_output_neurons: Optional[int] = None
    output_neurons: Optional[np.ndarray] = None
    membrane_potentials: Optional[np.ndarray] = None
    duration: Optional[int] = None
    current_amplitude: Optional[float] = None
    weight_variance: Optional[float] = None
    input_spike_times: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.uint8)
    )
    adjacency_matrix: Optional[spmatrix] = None

    def __post_init__(self):
        # === Basic type checks ===
        if self.connection_prob is not None:
            if not isinstance(self.connection_prob, (int, float)):
                raise TypeError("'connection_prob' must be a float or int.")
            if not (0 <= self.connection_prob <= 1):
                raise ValueError("'connection_prob' must be between 0 and 1.")

        if self.is_random_uniform is not None and not isinstance(
            self.is_random_uniform, bool
        ):
            raise TypeError("'is_random_uniform' must be a boolean if provided.")

        if self.num_neurons is not None:
            if not isinstance(self.num_neurons, int):
                raise TypeError("'num_neurons' must be an integer if provided.")
            if self.num_neurons <= 0:
                raise ValueError("'num_neurons' must be positive.")

        if self.num_output_neurons is not None:
            if not isinstance(self.num_output_neurons, int):
                raise TypeError("'num_output_neurons' must be an integer if provided.")
            if self.num_output_neurons <= 0:
                raise ValueError("'num_output_neurons' must be positive.")

        if self.output_neurons is not None:
            if not isinstance(self.output_neurons, np.ndarray):
                raise TypeError("'output_neurons' must be a NumPy array if provided.")
            if self.output_neurons.ndim != 1:
                raise ValueError("'output_neurons' must be a 1D NumPy array.")
            if not np.issubdtype(self.output_neurons.dtype, np.integer):
                raise TypeError("'output_neurons' must contain integers.")

        if not isinstance(self.membrane_threshold, (int, float)):
            raise TypeError("'membrane_threshold' must be a float or int.")

        if not isinstance(self.leak_coefficient, (int, float)):
            raise TypeError("'leak_coefficient' must be a float or int.")
        if not (0 <= self.leak_coefficient < 1):
            raise ValueError("'leak_coefficient' must be in the range [0, 1).")

        if not isinstance(self.refractory_period, int):
            raise TypeError("'refractory_period' must be an integer.")
        if self.refractory_period <= 0:
            raise ValueError("'refractory_period' must be positive.")

        if self.duration is not None:
            if not isinstance(self.duration, int):
                raise TypeError("'duration' must be an integer if provided.")
            if self.duration <= 0:
                raise ValueError("'duration' must be positive.")

        if self.current_amplitude is not None:
            if not isinstance(self.current_amplitude, (int, float)):
                raise TypeError(
                    "'current_amplitude' must be a float or int if provided."
                )

        if self.small_world_graph_p is not None:
            if not isinstance(self.small_world_graph_p, (int, float)):
                raise TypeError("'small_world_graph_p' must be a float.")
            if not (0 <= self.small_world_graph_p <= 1):
                raise ValueError("'small_world_graph_p' must be between 0 and 1.")

        if self.small_world_graph_k is not None:
            if not isinstance(self.small_world_graph_k, int):
                raise TypeError("'small_world_graph_k' must be an int.")
            if self.small_world_graph_k < 2 or self.small_world_graph_k % 2 != 0:
                raise ValueError(
                    "'small_world_graph_k' must be an even integer ≥ 2."
                )

        if self.weight_variance is not None:
            if not isinstance(self.weight_variance, (int, float)):
                raise TypeError("'weight_variance' must be a float or int if provided.")
            if self.weight_variance < 0:
                raise ValueError("'weight_variance' must be non-negative.")

        if not isinstance(self.input_spike_times, np.ndarray):
            raise TypeError("'input_spike_times' must be a NumPy array.")
        if self.input_spike_times.ndim != 2:
            raise ValueError("'input_spike_times' must be a 2D NumPy array.")
        if not np.issubdtype(self.input_spike_times.dtype, np.integer):
            raise TypeError("'input_spike_times' must contain integers.")

        if self.adjacency_matrix is not None:
            if not issparse(self.adjacency_matrix):
                raise TypeError("'adjacency_matrix' must be a SciPy sparse matrix.")
            if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
                raise ValueError("'adjacency_matrix' must be square.")
            if not np.issubdtype(self.adjacency_matrix.data.dtype, np.floating):
                raise TypeError("'adjacency_matrix' must contain float weights.")
            if (
                self.adjacency_matrix.shape[0]
                < self.input_spike_times.shape[0]
            ):
                raise ValueError(
                    "'adjacency_matrix' must accommodate all input neurons."
                )

        if self.membrane_potentials is not None:
            if not isinstance(self.membrane_potentials, np.ndarray):
                raise TypeError(
                    "'membrane_potentials' must be a NumPy array if provided."
                )
            if self.membrane_potentials.ndim != 1:
                raise ValueError("'membrane_potentials' must be a 1D NumPy array.")
            if not np.issubdtype(self.membrane_potentials.dtype, np.floating):
                raise TypeError("'membrane_potentials' must contain float values.")
            if self.membrane_threshold > 0:
                if (
                    np.any(self.membrane_potentials < 0)
                    or np.any(self.membrane_potentials > self.membrane_threshold)
                ):
                    raise ValueError(
                        "'membrane_potentials' must be in the range "
                        "[0, membrane_threshold]."
                    )
            elif self.membrane_threshold < 0:
                if (
                    np.any(self.membrane_potentials > 0)
                    or np.any(self.membrane_potentials < self.membrane_threshold)
                ):
                    raise ValueError(
                        "'membrane_potentials' must be in the range "
                        "[membrane_threshold, 0]."
                    )

        # === Mutual exclusivity checks ===
        if self.num_neurons is not None and self.adjacency_matrix is not None:
            raise ValueError(
                "Provide either 'num_neurons' OR 'adjacency_matrix', not both."
            )
        if self.num_neurons is None and self.adjacency_matrix is None:
            raise ValueError(
                "You must provide one of 'num_neurons' or 'adjacency_matrix'."
            )

        if self.num_neurons is not None and self.mean_weight is None:
            raise ValueError(
                "When 'num_neurons' is provided, 'mean_weight' must also be provided."
            )

        if self.adjacency_matrix is not None:
            if self.mean_weight is not None:
                raise ValueError(
                    "Do not provide 'mean_weight' when using 'adjacency_matrix'."
                )
            if self.weight_variance is not None:
                raise ValueError(
                    "Do not provide 'weight_variance' when using 'adjacency_matrix'."
                )

        # === Output neuron configuration ===
        if (self.num_output_neurons is None) == (self.output_neurons is None):
            raise ValueError(
                "Exactly one of 'num_output_neurons' or 'output_neurons' "
                "must be provided."
            )

        if self.output_neurons is not None and self.num_neurons is not None:
            if (
                np.any(self.output_neurons >= self.num_neurons)
                or np.any(self.output_neurons < 0)
            ):
                raise ValueError(
                    "'output_neurons' contains invalid indices (out of bounds)."
                )

        # === Connection type validation ===
        if self.is_random_uniform:
            if (
                self.small_world_graph_k is not None
                or self.small_world_graph_p is not None
            ):
                raise ValueError(
                    "'small_world_graph_k' and 'small_world_graph_p' must not be "
                    "provided when 'is_random_uniform' is True."
                )
            if self.connection_prob is None:
                raise ValueError(
                    "'connection_prob' must be provided when 'is_random_uniform' is True."
                )
        else:
            if self.connection_prob is not None:
                raise ValueError(
                    "'connection_prob' must not be provided when 'is_random_uniform' is False."
                )
            if (
                self.small_world_graph_k is None
                or self.small_world_graph_p is None
            ):
                raise ValueError(
                    "'small_world_graph_k' and 'small_world_graph_p' must be provided "
                    "when 'is_random_uniform' is False."
                )
            

def require_simulation_run(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.spike_matrix_output is None:
            raise ValueError("Simulation not yet run.")
        return method(self, *args, **kwargs)
    return wrapper


class SNN:
    """Implements a Spiking Neural Network (SNN) with customizable topology and simulation features."""

    def __init__(self, simulation_params: SimulationParams) -> None:
        self.simulation_params = simulation_params

        # === Configuration from simulation parameters ===
        self.current_amplitude = (
            simulation_params.membrane_threshold
            if simulation_params.current_amplitude is None
            else simulation_params.current_amplitude
        )

        self.num_input_neurons: int = simulation_params.input_spike_times.shape[0]
        self.membrane_threshold: float = simulation_params.membrane_threshold
        self.refractory_period: int = simulation_params.refractory_period
        self.leak_coefficient: float = simulation_params.leak_coefficient

        self.simulation_duration: int = (
            simulation_params.input_spike_times.shape[1]
            if simulation_params.duration is None
            else simulation_params.duration
        )
        self.input_spike_times = simulation_params.input_spike_times

        if simulation_params.num_neurons is not None:
            self.num_neurons = simulation_params.num_neurons
            self.weights_mean = simulation_params.mean_weight
            self.weights_variance = (
                simulation_params.weight_variance
                if simulation_params.weight_variance is not None
                else 0.1 * self.weights_mean
            )

            if (
                simulation_params.is_random_uniform is not None
                and not simulation_params.is_random_uniform
            ):
                self._generate_synaptic_weights_small_world(
                    simulation_params.small_world_graph_p,
                    simulation_params.small_world_graph_k,
                )
            else:
                self._generate_synaptic_weights_random_uniform(
                    simulation_params.connection_prob
                )

        elif simulation_params.adjacency_matrix is not None:
            self.synaptic_weights = simulation_params.adjacency_matrix.copy()
            self.num_neurons = self.synaptic_weights.shape[0]
            in_degrees = self.synaptic_weights.getnnz(axis=0)
            self.mean_in_degree = in_degrees.mean()

            weights_array = self.synaptic_weights.data
            if weights_array.size == 0:
                raise ValueError("Provided adjacency matrix has no non-zero weights.")
            self.weights_mean = float(np.mean(weights_array))
            self.weights_variance = float(np.var(weights_array))

        else:
            raise ValueError(
                "You must provide either 'num_neurons' or 'adjacency_matrix'."
            )

        if simulation_params.num_output_neurons is not None:
            self.num_output_neurons: int = simulation_params.num_output_neurons
            hidden_indices = np.arange(self.num_input_neurons, self.num_neurons)
            if len(hidden_indices) < self.num_output_neurons:
                raise ValueError("Not enough hidden neurons to select output neurons.")
            self.output_neurons = np.random.choice(
                hidden_indices,
                size=self.num_output_neurons,
                replace=False,
            )
        else:
            self.output_neurons = simulation_params.output_neurons
            self.num_output_neurons = len(simulation_params.output_neurons)

        self.time_step: int = 1

        # === Internal state ===
        self.tot_spikes: int = 0
        if simulation_params.membrane_potentials is None:
            self.membrane_potentials: np.ndarray = np.random.uniform(
                0, self.membrane_threshold, self.num_neurons
            )
        else:
            if len(simulation_params.membrane_potentials) != self.num_neurons:
                raise ValueError(
                    "Length of 'membrane_potentials' must be equal to 'num_neurons'."
                )
            self.membrane_potentials = simulation_params.membrane_potentials

        self.membrane_potentials_init = self.membrane_potentials.copy()
        self.spike_matrix: Optional[np.ndarray] = None
        self.spike_matrix_output: Optional[np.ndarray] = None
        self.refractory_timer: np.ndarray = np.zeros(self.num_neurons)

    def _generate_synaptic_weights_small_world(
        self,
        small_world_graph_p: float = 0.1,
        small_world_graph_k: int = 10,
    ) -> None:
        """Generate synaptic weights using a small-world graph."""
        small_world_graph = nx.watts_strogatz_graph(
            n=self.num_neurons,
            k=small_world_graph_k,
            p=small_world_graph_p,
            seed=None,
        )
        synaptic_weights = np.zeros((self.num_neurons, self.num_neurons))

        for i, j in small_world_graph.edges():
            if np.random.rand() < 0.5:
                synaptic_weights[i, j] = np.random.normal(
                    loc=self.weights_mean,
                    scale=abs(self.weights_mean) * self.weights_variance,
                )
            else:
                synaptic_weights[j, i] = np.random.normal(
                    loc=self.weights_mean,
                    scale=abs(self.weights_mean) * self.weights_variance,
                )

        np.fill_diagonal(synaptic_weights, 0)
        self.synaptic_weights = csr_matrix(synaptic_weights)

        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

    def _generate_synaptic_weights_random_uniform(
        self,
        connection_prob: float = 0.2,
    ) -> None:
        """Generate synaptic weights using random uniform connections (vectorized)."""
        n = self.num_neurons
        connection_mask = np.random.rand(n, n) < connection_prob
        np.fill_diagonal(connection_mask, False)

        weights = np.zeros((n, n))
        weights[connection_mask] = np.random.normal(
            loc=self.weights_mean,
            scale=abs(self.weights_mean) * self.weights_variance,
            size=np.count_nonzero(connection_mask),
        )
        self.synaptic_weights = csr_matrix(weights)

        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

    @property
    def avg_in_degree(self):
        return self.synaptic_weights.getnnz(axis=0).mean()

    def simulate(self) -> Optional[np.ndarray]:
        """Run the optimized simulation of the SNN."""
        if self.input_spike_times.shape[0] > self.num_neurons:
            warnings.warn(
                "Number of input spike rows exceeds 'num_neurons'. "
                "Simulation may behave unexpectedly.",
                category=UserWarning
            )

        if np.any(self.output_neurons >= self.num_neurons) or np.any(self.output_neurons < 0):
            warnings.warn(
                "Provided output neuron indices are out of bounds. "
                "Simulation may behave unexpectedly.",
                category=UserWarning
            )

        if len(self.membrane_potentials) != self.num_neurons:
            warnings.warn(
                "Length of 'membrane_potentials' is not equal to 'num_neurons'. "
                "Simulation may behave unexpectedly.",
                category=UserWarning
            )

        if self.synaptic_weights.data.size == 0:
            warnings.warn(
                "Provided adjacency matrix has no non-zero weights. "
                "Simulation may behave unexpectedly.",
                category=UserWarning
            )

        self.tot_spikes = 0
        self.spike_matrix = np.zeros(
            (self.simulation_duration, self.num_neurons),
            dtype=np.int8
        )

        for t in range(self.simulation_duration):
            self.refractory_timer -= self.time_step
            np.clip(self.refractory_timer, 0, None, out=self.refractory_timer)

            if t < self.input_spike_times.shape[1]:
                spike_vector = self.input_spike_times[:, t]
                input_neurons = np.where(spike_vector == 1)[0]
                self.membrane_potentials[input_neurons] += self.membrane_threshold

            spiking_neurons = (
                (self.membrane_potentials >= self.membrane_threshold) &
                (self.refractory_timer == 0)
            )

            self.spike_matrix[t, :] = spiking_neurons
            self.tot_spikes += np.count_nonzero(spiking_neurons)

            self.membrane_potentials[spiking_neurons] = 0
            self.refractory_timer[spiking_neurons] = self.refractory_period + 1

            input_vector = spiking_neurons.astype(np.float32)
            synaptic_input = self.synaptic_weights.T.dot(input_vector)
            self.membrane_potentials *= (1.0 - self.leak_coefficient)
            self.membrane_potentials += synaptic_input

        self.spike_matrix_output = self.spike_matrix[:, self.output_neurons]
        return self.spike_matrix_output

    @require_simulation_run
    def get_spike_time_lists_output(self) -> List[List[int]]:
        """Return a list of lists with spike times for each output neuron."""
        return [
            list(np.where(self.spike_matrix_output[:, i] == 1)[0])
            for i in range(self.num_output_neurons)
        ]


    @require_simulation_run
    def get_spike_counts(self) -> np.ndarray:
        """Total spike count per output neuron."""
        return np.sum(self.spike_matrix_output, axis=0)


    @require_simulation_run
    def get_spike_variances(self) -> np.ndarray:
        """Variance of spike sequences per output neuron."""
        return np.var(self.spike_matrix_output, axis=0)


    @require_simulation_run
    def get_first_spike_times(self) -> np.ndarray:
        """First spike time per output neuron."""
        has_spike = np.any(self.spike_matrix_output == 1, axis=0)
        first_spike_times = np.argmax(self.spike_matrix_output == 1, axis=0)
        return np.where(has_spike, first_spike_times, -1)


    @require_simulation_run
    def get_mean_spike_times(self) -> np.ndarray:
        """Mean spike time per output neuron."""
        spike_counts = self.get_spike_counts()
        times = np.arange(self.spike_matrix_output.shape[0])[:, None]
        weighted_times = self.spike_matrix_output * times
        sum_times = np.sum(weighted_times, axis=0)

        mean_spike_times = np.full_like(spike_counts, fill_value=-1.0, dtype=float)
        nonzero_mask = spike_counts > 0
        mean_spike_times[nonzero_mask] = (
            sum_times[nonzero_mask] / spike_counts[nonzero_mask]
        )
        return mean_spike_times

    @require_simulation_run
    def get_last_spike_times(self) -> np.ndarray:
        """Last spike time per output neuron."""
        has_spike = np.any(self.spike_matrix_output == 1, axis=0)
        last_spike_times = (
            self.spike_matrix_output.shape[0] - 1
            - np.argmax(self.spike_matrix_output[::-1] == 1, axis=0)
        )
        return np.where(has_spike, last_spike_times, -1)

    @require_simulation_run
    def get_mean_isi_per_neuron(self) -> np.ndarray:
        """Mean ISI (inter-spike interval) per output neuron."""
        mean_isis = np.full(self.num_output_neurons, -1.0, dtype=float)
        for i in range(self.num_output_neurons):
            spike_times = np.where(self.spike_matrix_output[:, i] == 1)[0]
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                mean_isis[i] = np.mean(isis)
        return mean_isis

    @require_simulation_run
    def get_isi_variance_per_neuron(self) -> np.ndarray:
        """ISI variance per output neuron."""
        isi_vars = np.full(self.num_output_neurons, -1.0, dtype=float)
        for i in range(self.num_output_neurons):
            spike_times = np.where(self.spike_matrix_output[:, i] == 1)[0]
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                isi_vars[i] = np.var(isis)
        return isi_vars

    @require_simulation_run
    def extract_features_from_spikes(self) -> dict:
        """Extract all key features from output neurons as a dictionary."""
        return {
            "spike_counts": self.get_spike_counts(),
            "spike_variances": self.get_spike_variances(),
            "mean_spike_times": self.get_mean_spike_times(),
            "first_spike_times": self.get_first_spike_times(),
            "last_spike_times": self.get_last_spike_times(),
            "mean_isi": self.get_mean_isi_per_neuron(),
            "isi_variances": self.get_isi_variance_per_neuron(),
        }
        
    def save_topology(self, filename: str = DEFAULT_MATRIX_PATH) -> None:
        """Save synaptic weights matrix to .npz file."""
        save_npz(filename, self.synaptic_weights)
        print(f"✅ Matrices saved to '{filename}'")


    def load_topology(self, filename: str = DEFAULT_MATRIX_PATH) -> None:
        """Load synaptic weights matrix from .npz file."""
        self.synaptic_weights = load_npz(filename)
        self.num_neurons = self.synaptic_weights.shape[0]
        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

        weights_array = self.synaptic_weights.data
        self.weights_mean = float(np.mean(weights_array))
        self.weights_variance = float(np.var(weights_array))

    def set_topology(self, topology) -> None:
        """Set synaptic weights matrix."""
        self.synaptic_weights = topology.copy()
        self.num_neurons = self.synaptic_weights.shape[0]
        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

        weights_array = self.synaptic_weights.data
        self.weights_mean = float(np.mean(weights_array))
        self.weights_variance = float(np.var(weights_array))

    def get_topology(self) -> csr_matrix:
        return self.synaptic_weights.copy()


    def save_membrane_potentials(
        self,
        filename: str = DEFAULT_POTENTIALS_PATH
    ) -> None:
        """Save membrane potentials to .npy file."""
        np.save(filename, self.membrane_potentials)


    def load_membrane_potentials(
        self,
        filename: str = DEFAULT_POTENTIALS_PATH
    ) -> None:
        """Load membrane potentials from .npy file."""
        self.membrane_potentials = np.load(filename)


    def set_membrane_potentials(self, membrane_potentials: np.ndarray) -> None:
        """Set membrane potentials from provided array."""
        self.membrane_potentials = membrane_potentials.copy()


    def get_membrane_potentials(self) -> np.ndarray:
        return self.membrane_potentials.copy()


    def save_output_neurons(self, filename: str = DEFAULT_OUTPUT_NEURONS_PATH) -> None:
        """Save selected output neuron indices to a .npy file."""
        np.save(filename, self.output_neurons)


    def load_output_neurons(self, filename: str = DEFAULT_OUTPUT_NEURONS_PATH) -> None:
        """Load selected output neuron indices from a .npy file."""
        self.output_neurons = np.load(filename)
        self.num_output_neurons = len(self.output_neurons)


    def set_output_neurons(self, indices: np.ndarray) -> None:
        """Manually set the selected output neuron indices."""
        self.output_neurons = indices


    def get_output_neurons(self) -> np.ndarray:
        return self.output_neurons


    def set_input_spike_times(self, input_spike_times: np.ndarray) -> None:
        """Set input spike times and adjust simulation duration."""
        self.input_spike_times = input_spike_times
        self.simulation_duration = (
            int(np.max(input_spike_times)) + int(self.num_neurons / 10)
        )

    def calculate_mean_isi(self) -> float:
        """Compute mean inter-spike interval (ISI)."""
        spike_times_list = self.get_spike_time_lists_output()
        total_intervals = []
        for spike_times in spike_times_list:
            if len(spike_times) > 1:
                intervals = np.diff(spike_times)
                total_intervals.extend(intervals)
        if total_intervals:
            return float(np.mean(total_intervals))
        return float(self.simulation_duration)


    def reset_synaptic_weights(self, mean: float, std: Optional[float] = None) -> None:
        """Reset synaptic weights with new normal distribution."""
        if not hasattr(self, "synaptic_weights"):
            raise ValueError("Synaptic weights not initialized.")

        if std is None:
            std = 0.1

        rows, cols = self.synaptic_weights.nonzero()
        new_weights = np.random.normal(loc=mean, scale=std * mean, size=len(rows))

        self.synaptic_weights = csr_matrix(
            (new_weights, (rows, cols)),
            shape=self.synaptic_weights.shape
        )
        self.weights_mean = mean
        self.weights_variance = (std * mean) ** 2

    def get_network_parameters(self) -> dict:
        """Return key parameters of the spiking neural network."""
        return {
            "num_neurons": self.num_neurons,
            "num_input_neurons": self.num_input_neurons,
            "num_output_neurons": self.num_output_neurons,
            "output_neurons": (
                self.output_neurons.tolist()
                if hasattr(self.output_neurons, "tolist")
                else self.output_neurons
            ),
            "membrane_threshold": self.membrane_threshold,
            "refractory_period": self.refractory_period,
            "leak_coefficient": self.leak_coefficient,
            "simulation_duration": self.simulation_duration,
            "weights_mean": self.weights_mean,
            "weights_variance": self.weights_variance,
            "avg_in_degree": self.mean_in_degree,
            "time_step": self.time_step,
            "current_amplitude": self.current_amplitude
        }

    def reset(self) -> None:
        """
        Reset internal simulation state: membrane potentials, spike matrices, refractory timers.
        Does not reset the synaptic weights or topology.
        """
        self.tot_spikes = 0
        self.spike_matrix = None
        self.spike_matrix_output = None
        self.refractory_timer = np.zeros_like(self.membrane_potentials)
        self.membrane_potentials = self.membrane_potentials_init.copy()


def load_output_neurons(filename: str = DEFAULT_OUTPUT_NEURONS_PATH) -> np.ndarray:
    """Load selected output neuron indices from a .npy file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ File '{filename}' not found.")
    return np.load(filename)


def load_membrane_potentials(filename: str = DEFAULT_POTENTIALS_PATH) -> np.ndarray:
    """Load membrane potentials from a .npy file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ File '{filename}' not found.")
    return np.load(filename)


def load_topology(filename: str = DEFAULT_MATRIX_PATH) -> csr_matrix:
    """Load synaptic weights matrix from a .npz file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ File '{filename}' not found.")
    return load_npz(filename)
