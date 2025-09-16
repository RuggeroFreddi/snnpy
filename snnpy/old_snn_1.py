"""Spiking Neural Network (SNN) model."""

from dataclasses import dataclass, field
from functools import wraps
import os
from typing import Optional, List
import warnings

import networkx as nx  
import numpy as np
from scipy.ndimage import label
from scipy.sparse import (
    csr_matrix,
    issparse,
    load_npz,
    save_npz,
    spmatrix,
)

DEFAULT_MATRIX_PATH = "dati/snn_matrices.npz"
DEFAULT_POTENTIALS_PATH = "dati/membrane_potentials.npy"
DEFAULT_OUTPUT_NEURONS_PATH = "dati/output_neurons.npy"


@dataclass
class STDPParams:
    enabled: bool = True
    tau_plus: float = 0.02
    tau_minus: float = 0.04
    A_plus: float = 1e-3
    A_minus: Optional[float] = None
    eta: float = 1.0
    W_max: float = 1.0
    clip: bool = True
    nearest_neighbor: bool = False
    lock_A_minus: bool = False

    def __post_init__(self):
        for name, val in [
            ("enabled", self.enabled),
            ("clip", self.clip),
            ("nearest_neighbor", self.nearest_neighbor),
            ("lock_A_minus", self.lock_A_minus),
        ]:
            if not isinstance(val, bool):
                raise TypeError(f"'{name}' must be a bool.")

        for name, val in [
            ("tau_plus", self.tau_plus),
            ("tau_minus", self.tau_minus),
            ("A_plus", self.A_plus),
            ("eta", self.eta),
            ("W_max", self.W_max),
        ]:
            if not isinstance(val, (int, float)):
                raise TypeError(f"'{name}' must be a float or int.")
            if not np.isfinite(val):
                raise ValueError(f"'{name}' must be finite.")

        if self.tau_plus <= 0:
            raise ValueError("'tau_plus' must be > 0.")
        if self.tau_minus <= 0:
            raise ValueError("'tau_minus' must be > 0.")
        if self.A_plus < 0:
            raise ValueError("'A_plus' must be >= 0.")
        if self.eta < 0:
            raise ValueError("'eta' must be >= 0.")
        if self.W_max <= 0:
            raise ValueError("'W_max' must be > 0.")

        if self.lock_A_minus:
            if self.A_minus is not None:
                raise ValueError(
                    "With lock_A_minus=True, 'A_minus' must not be provided. "
                    "It is derived as A_plus * (tau_plus / tau_minus)."
                )
            self.A_minus = float(self.A_plus) * (
                float(self.tau_plus) / float(self.tau_minus)
            )
        else:
            if self.A_minus is None:
                self.A_minus = 1e-3

            if not isinstance(self.A_minus, (int, float)):
                raise TypeError("'A_minus' must be a float or int.")
            if not np.isfinite(self.A_minus):
                raise ValueError("'A_minus' must be finite.")
            if self.A_minus < 0:
                raise ValueError("'A_minus' must be >= 0.")


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
            raise ValueError("Exactly one of 'num_output_neurons' or 'output_neurons' must be provided.")

        if self.output_neurons is not None:
            n = self.num_neurons if self.num_neurons is not None else (
                self.adjacency_matrix.shape[0] if self.adjacency_matrix is not None else None
            )
            if n is not None:
                if np.any(self.output_neurons < 0) or np.any(self.output_neurons >= n):
                    raise ValueError("'output_neurons' contains invalid indices (out of bounds).")

        # === Connection type validation ===
        if self.num_neurons is not None:
            if self.is_random_uniform is True:
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

    def __init__(self, simulation_params: SimulationParams, stdp_params: Optional[STDPParams] = None) -> None:
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

            if simulation_params.is_random_uniform is True:
                self._generate_synaptic_weights_random_uniform(
                    simulation_params.connection_prob
                )
            else:
                self._generate_synaptic_weights_small_world(
                    simulation_params.small_world_graph_p,
                    simulation_params.small_world_graph_k,
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

        self.stdp = stdp_params
        if self.stdp and self.stdp.enabled:
            self._init_stdp()
        self._recompute_input_mean_current()

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
        self.synaptic_weights.sort_indices()


        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)

    def _generate_synaptic_weights_random_uniform(
        self,
        connection_prob: float = 0.2,
    ) -> None:
        """Generate synaptic weights using random uniform connections (vectorized)."""
        num_neurons = self.num_neurons
        connection_mask = np.random.rand(num_neurons, num_neurons) < connection_prob
        np.fill_diagonal(connection_mask, False)

        weights = np.zeros((num_neurons, num_neurons))
        weights[connection_mask] = np.random.normal(
            loc=self.weights_mean,
            scale=abs(self.weights_mean) * self.weights_variance,
            size=np.count_nonzero(connection_mask),
        )
        self.synaptic_weights = csr_matrix(weights)
        self.synaptic_weights.sort_indices()


        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)

    @property
    def avg_in_degree(self):
        return self.synaptic_weights.getnnz(axis=0).mean()

    def _init_stdp(self):
        """Initialize STDP traces, decays, and in-neighbor structures."""
        self.trace_pre = np.zeros(self.num_neurons, dtype=np.float32)
        self.trace_post = np.zeros(self.num_neurons, dtype=np.float32)
        self._decay_pre = float(np.exp(-self.time_step / self.stdp.tau_plus))
        self._decay_post = float(np.exp(-self.time_step / self.stdp.tau_minus))
        self._build_in_neighbors()


    def _build_in_neighbors(self):
        """Precompute in-neighbors and CSR data positions for each postsynaptic column."""
        weight_matrix: csr_matrix = self.synaptic_weights.tocsr()
        row_pointer, col_indices = weight_matrix.indptr, weight_matrix.indices
        num_columns = weight_matrix.shape[1]

        in_neighbors = [[] for _ in range(num_columns)]
        in_data_positions = [[] for _ in range(num_columns)]

        for pre_neuron in range(weight_matrix.shape[0]):
            row_start, row_end = row_pointer[pre_neuron], row_pointer[pre_neuron + 1]
            row_columns = col_indices[row_start:row_end]
            # For each synapse pre_neuron -> post_neuron, store pre id and index in CSR.data
            for offset, post_neuron in enumerate(row_columns):
                in_neighbors[post_neuron].append(pre_neuron)
                in_data_positions[post_neuron].append(row_start + offset)

        self.in_neigh = [np.array(pre_list, dtype=np.int32) for pre_list in in_neighbors]
        self.in_pos = [np.array(pos_list, dtype=np.int32) for pos_list in in_data_positions]


    def _stdp_decay_traces(self):
        """Decay pre/post traces and optionally clip for nearest-neighbor STDP."""
        self.trace_pre *= self._decay_pre
        self.trace_post *= self._decay_post
        if self.stdp.nearest_neighbor:
            np.minimum(self.trace_pre, 1.0, out=self.trace_pre)
            np.minimum(self.trace_post, 1.0, out=self.trace_post)

    def _stdp_on_pre(self, pre_spike_indices: np.ndarray):
        """Handle PRE spikes: apply vectorized LTD on edges j->i (CSR row batches)."""
        if pre_spike_indices.size == 0:
            return
        W = self.synaptic_weights
        data = W.data
        indptr = W.indptr
        indices = W.indices
        c = (self.stdp.eta * self.stdp.A_minus) / self.stdp.W_max
        wmax = self.stdp.W_max
        clip = self.stdp.clip
        post_tr = self.trace_post
        for j in pre_spike_indices:
            s, e = indptr[j], indptr[j + 1]
            if s == e:
                continue
            cols = indices[s:e]
            w = data[s:e]
            np.multiply(w, 1.0 - c * post_tr[cols], out=w)
            if clip:
                np.clip(w, 0.0, wmax, out=w)

    def _stdp_on_post(self, post_spike_indices: np.ndarray):
        """Handle POST spikes: apply vectorized LTP on edges j->i using precomputed in_pos."""
        if post_spike_indices.size == 0:
            return
        W = self.synaptic_weights
        data = W.data
        alpha = self.stdp.eta * self.stdp.A_plus
        inv_wmax = 1.0 / self.stdp.W_max
        wmax = self.stdp.W_max
        clip = self.stdp.clip
        pre_tr = self.trace_pre
        in_pos = self.in_pos
        in_neigh = self.in_neigh
        for i in post_spike_indices:
            idxs = in_pos[i]
            if idxs.size == 0:
                continue
            w = data[idxs]
            pre = pre_tr[in_neigh[i]]
            w = w + alpha * pre * (1.0 - w * inv_wmax)
            if clip:
                np.clip(w, 0.0, wmax, out=w)
            data[idxs] = w

    def disable_stdp(self) -> None:
        """Disable STDP and clear its internal buffers."""
        if self.stdp is None:
            return
        self.stdp.enabled = False
        # Drop STDP state to free memory and avoid accidental use
        for attr in (
            "trace_pre",
            "trace_post",
            "in_neigh",
            "in_pos",
            "_decay_pre",
            "_decay_post",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _recompute_input_mean_current(self) -> None:
        """Compute mean input current per neuron per time step (I)."""
        if self.input_spike_times.size == 0:
            self.input_mean_current = 0.0
        else:
            num_input_spikes = int(np.count_nonzero(self.input_spike_times))
            self.input_mean_current = float(num_input_spikes) / float(self.input_spike_times.shape[0] * self.input_spike_times.shape[1])

    def simulate(self) -> Optional[np.ndarray]:
        """Run the optimized simulation of the SNN."""
        # --- sanity checks (lasciati invariati) ---
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

        T = self.simulation_duration
        N = self.num_neurons
        Nin = self.num_input_neurons
        inputs = self.input_spike_times
        inputs_T = inputs.shape[1] if inputs.size else 0
        mem = self.membrane_potentials
        refr = self.refractory_timer
        out_idx = self.output_neurons
        stdp = self.stdp
        leak_factor = np.float32(1.0 - self.leak_coefficient)
        curr_amp = np.float32(self.current_amplitude)

        if mem.dtype != np.float32:
            mem = mem.astype(np.float32, copy=False)
            self.membrane_potentials = mem
        if refr.dtype != np.int32 and refr.dtype != np.int16 and refr.dtype != np.float32:
            refr = refr.astype(np.float32, copy=False)
            self.refractory_timer = refr

        W = self.synaptic_weights.tocsr()
        indptr, indices, data = W.indptr, W.indices, W.data

        self.tot_spikes = 0
        self.spike_matrix = np.zeros((T, N), dtype=np.int8)
        spikes_out = self.spike_matrix  # alias

        for t in range(T):
            refr -= self.time_step
            np.clip(refr, 0, None, out=refr)
            if t < inputs_T:
                spikes_t = inputs[:, t] 
                mem[:Nin] += curr_amp * spikes_t  

            spiking_mask = (mem >= self.membrane_threshold) & (refr == 0)
            spikes_out[t, :] = spiking_mask

            if not spiking_mask.any():
                mem *= leak_factor
                continue

            spk_idx = np.flatnonzero(spiking_mask)
            self.tot_spikes += spk_idx.size

            if stdp and stdp.enabled:
                self._stdp_decay_traces()
                self._stdp_on_pre(spk_idx)
                self._stdp_on_post(spk_idx)
                self.trace_pre[spk_idx] += 1.0
                self.trace_post[spk_idx] += 1.0
                if stdp.nearest_neighbor:
                    np.minimum(self.trace_pre, 1.0, out=self.trace_pre)
                    np.minimum(self.trace_post, 1.0, out=self.trace_post)

            mem[spiking_mask] = 0.0
            refr[spiking_mask] = self.refractory_period + 1

            mem *= leak_factor

            for j in spk_idx:
                start, end = indptr[j], indptr[j + 1]
                if start != end:
                    cols = indices[start:end]
                    mem[cols] += data[start:end]

        self.spike_matrix_output = self.spike_matrix[:, out_idx]
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
    def get_burst_counts(self) -> np.ndarray:
        """Return the number of spike bursts per output neuron."""
        burst_counts = np.zeros(self.num_output_neurons, dtype=int)
        for i in range(self.num_output_neurons):
            spike_train = self.spike_matrix_output[:, i]
            _, num_bursts = label(spike_train)
            burst_counts[i] = num_bursts
        return burst_counts
    
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
            "burst_counts": self.get_burst_counts(),  
        }
        
    def save_topology(self, filename: str = DEFAULT_MATRIX_PATH) -> None:
        """Save synaptic weights matrix to .npz file."""
        save_npz(filename, self.synaptic_weights)

    def load_topology(self, filename: str = DEFAULT_MATRIX_PATH) -> None:
        """Load synaptic weights matrix from a .npz file."""
        weight_matrix = load_npz(filename).tocsr()
        weight_matrix.sort_indices()
        self.synaptic_weights = weight_matrix

        if self.stdp and self.stdp.enabled:
            self._build_in_neighbors()

        self.num_neurons = self.synaptic_weights.shape[0]
        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

        weights_array = self.synaptic_weights.data
        self.weights_mean = float(np.mean(weights_array))
        self.weights_variance = float(np.var(weights_array))
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)

    def set_topology(self, topology) -> None:
        """Set synaptic weights matrix."""
        weight_matrix = topology.tocsr(copy=True) if issparse(topology) else csr_matrix(topology)
        weight_matrix.sort_indices()
        self.synaptic_weights = weight_matrix

        if self.stdp and self.stdp.enabled:
            self._build_in_neighbors()

        self.num_neurons = weight_matrix.shape[0]
        in_degrees = weight_matrix.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

        weights_array = weight_matrix.data
        self.weights_mean = float(np.mean(weights_array))
        self.weights_variance = float(np.var(weights_array))
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)

    def get_topology(self) -> csr_matrix:
        return self.synaptic_weights.copy()

    def save_membrane_potentials(
        self,
        filename: str = DEFAULT_POTENTIALS_PATH
    ) -> None:
        """Save membrane potentials to .npy file."""
        np.save(filename, self.membrane_potentials)

    def load_membrane_potentials(self, filename: str = DEFAULT_POTENTIALS_PATH) -> None:
        """Load membrane potentials from .npy file and validate them."""
        arr = np.load(filename)
        arr = np.asarray(arr)

        if arr.ndim != 1:
            raise ValueError("'membrane_potentials' must be a 1D array.")

        n = getattr(self, "num_neurons", None)
        if n is None and hasattr(self, "synaptic_weights"):
            n = self.synaptic_weights.shape[0]
        if n is not None and arr.size != n:
            raise ValueError(
                f"'membrane_potentials' length ({arr.size}) does not match num_neurons ({n})."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError("'membrane_potentials' must be finite values.")

        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)

        thr = float(self.membrane_threshold)
        if thr > 0:
            if (arr < 0).any() or (arr > thr).any():
                raise ValueError("'membrane_potentials' must be in [0, membrane_threshold].")
        elif thr < 0:
            if (arr > 0).any() or (arr < thr).any():
                raise ValueError("'membrane_potentials' must be in [membrane_threshold, 0].")

        self.membrane_potentials = arr.astype(np.float32, copy=True)
        self.membrane_potentials_init = self.membrane_potentials.copy()

        if not hasattr(self, "refractory_timer") or self.refractory_timer.shape[0] != arr.size:
            self.refractory_timer = np.zeros(arr.size, dtype=np.float32)

    def set_membrane_potentials(self, membrane_potentials: np.ndarray) -> None:
        """Set membrane potentials from provided array with validation."""
        arr = np.asarray(membrane_potentials)

        if arr.ndim != 1:
            raise ValueError("'membrane_potentials' must be a 1D array.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("'membrane_potentials' must be finite values.")

        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)

        n = getattr(self, "num_neurons", None)
        if n is None and hasattr(self, "synaptic_weights"):
            n = self.synaptic_weights.shape[0]
        if n is not None and arr.size != n:
            raise ValueError(
                f"'membrane_potentials' length ({arr.size}) does not match num_neurons ({n})."
            )

        thr = float(self.membrane_threshold)
        if thr > 0:
            if (arr < 0).any() or (arr > thr).any():
                raise ValueError("'membrane_potentials' must be in [0, membrane_threshold].")
        elif thr < 0:
            if (arr > 0).any() or (arr < thr).any():
                raise ValueError("'membrane_potentials' must be in [membrane_threshold, 0].")

        self.membrane_potentials = arr.astype(np.float32, copy=True)
        self.membrane_potentials_init = self.membrane_potentials.copy()

        if not hasattr(self, "refractory_timer") or self.refractory_timer.shape[0] != arr.size:
            self.refractory_timer = np.zeros(arr.size, dtype=np.float32)

    def get_membrane_potentials(self) -> np.ndarray:
        return self.membrane_potentials.copy()


    def save_output_neurons(self, filename: str = DEFAULT_OUTPUT_NEURONS_PATH) -> None:
        """Save selected output neuron indices to a .npy file."""
        np.save(filename, self.output_neurons)

    def load_output_neurons(self, filename: str = DEFAULT_OUTPUT_NEURONS_PATH) -> None:
        arr = np.load(filename)
        arr = np.asarray(arr)

        if arr.ndim != 1:
            raise ValueError("'output_neurons' must be a 1D array.")
        if not np.issubdtype(arr.dtype, np.integer):
            if np.all(np.isfinite(arr)) and np.all(np.mod(arr, 1) == 0):
                arr = arr.astype(np.int64)
            else:
                raise TypeError("'output_neurons' must contain integer indices.")

        n = getattr(self, "num_neurons", None)
        if n is None and hasattr(self, "synaptic_weights"):
            n = self.synaptic_weights.shape[0]
        if n is not None:
            if np.any(arr < 0) or np.any(arr >= n):
                raise ValueError("'output_neurons' are out of bounds.")

        self.output_neurons = arr
        self.num_output_neurons = int(arr.size)

    def set_output_neurons(self, indices: np.ndarray) -> None:
        arr = np.asarray(indices)

        if arr.ndim != 1:
            raise ValueError("'output_neurons' must be a 1D array.")
        if not np.issubdtype(arr.dtype, np.integer):
            if np.all(np.isfinite(arr)) and np.all(np.mod(arr, 1) == 0):
                arr = arr.astype(np.int64)
            else:
                raise TypeError("'output_neurons' must contain integer indices.")

        n = getattr(self, "num_neurons", None)
        if n is None and hasattr(self, "synaptic_weights"):
            n = self.synaptic_weights.shape[0]
        if n is not None:
            if np.any(arr < 0) or np.any(arr >= n):
                raise ValueError("'output_neurons' are out of bounds.")

        self.output_neurons = arr
        self.num_output_neurons = int(arr.size)

    def get_output_neurons(self) -> np.ndarray:
        return self.output_neurons

    def set_input_spike_times(self, input_spike_times: np.ndarray) -> None:
        """Set input spike times and adjust simulation duration."""
        self.input_spike_times = input_spike_times
        self.simulation_duration = (
            self.simulation_params.duration
            if self.simulation_params.duration is not None
            else input_spike_times.shape[1]
        )
        self._recompute_input_mean_current()

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
    
    def rescale_synaptic_weights_to_mean(self, target_mean: float) -> None:
        """Scale non-zero synaptic weights so their mean equals target_mean."""
        if not hasattr(self, "synaptic_weights"):
            raise ValueError("Synaptic weights not initialized.")
        if not np.isfinite(target_mean):
            raise ValueError("'target_mean' must be finite.")

        data = self.synaptic_weights.data
        if data.size == 0:
            raise ValueError("Synaptic weights matrix has no non-zero entries.")

        current_mean = float(np.mean(data))
        if current_mean == 0.0:
            if target_mean == 0.0:
                self.weights_mean = 0.0
                self.weights_variance = float(np.var(data))
                return
            raise ValueError("Cannot rescale from zero current mean to a non-zero target.")

        scale = float(target_mean) / current_mean
        self.synaptic_weights.data *= scale 
        weights_array = self.synaptic_weights.data
        self.weights_mean = float(np.mean(weights_array))
        self.weights_variance = float(np.var(weights_array))

    def reset_synaptic_weights(self, mean: float, std: Optional[float] = None) -> None:
        """Reset synaptic weights with new normal distribution."""
        if not hasattr(self, "synaptic_weights"):
            raise ValueError("Synaptic weights not initialized.")

        if std is None:
            std = 0.1

        old_weight_matrix = self.synaptic_weights.tocsr()
        rows, cols = old_weight_matrix.nonzero()
        new_weights = np.random.normal(loc=mean, scale=std * mean, size=rows.size)

        new_weight_matrix = csr_matrix((new_weights, (rows, cols)), shape=old_weight_matrix.shape)
        new_weight_matrix.sort_indices()               
        self.synaptic_weights = new_weight_matrix
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)

        if self.stdp and self.stdp.enabled:
            self._build_in_neighbors()

        self.weights_mean = mean
        self.weights_variance = (std * mean) ** 2

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

    def prune(self, fraction: float) -> None:
        """Prune the weakest fraction of synapses (by absolute weight)."""
        if not isinstance(fraction, (int, float)):
            raise TypeError("'fraction' must be a number in (0, 1).")
        fraction = float(fraction)
        if not (0.0 < fraction < 1.0):
            raise ValueError("'fraction' must be strictly between 0 and 1.")

        weight_matrix = self.synaptic_weights
        num_nonzeros = weight_matrix.nnz
        if num_nonzeros == 0:
            return

        # Number of synapses to prune (at least 1 and at most nnz - 1)
        num_to_prune = int(np.floor(fraction * num_nonzeros))
        num_to_prune = max(1, min(num_nonzeros - 1, num_to_prune))

        # Indices of the k smallest |w|
        abs_weights = np.abs(weight_matrix.data)
        prune_indices = np.argpartition(abs_weights, num_to_prune - 1)[:num_to_prune]

        # Zero-out and compact
        weight_matrix.data[prune_indices] = 0.0
        weight_matrix.eliminate_zeros()
        weight_matrix.sort_indices()
        weight_matrix.data = weight_matrix.data.astype(np.float32, copy=False)

        # Update consistent statistics
        if weight_matrix.nnz > 0:
            nonzero_weights = weight_matrix.data
            self.weights_mean = float(nonzero_weights.mean())  # mean over non-zero
            self.weights_variance = float(nonzero_weights.var())
        else:
            self.weights_mean = 0.0
            self.weights_variance = 0.0

        in_deg = weight_matrix.getnnz(axis=0)
        self.mean_in_degree = float(in_deg.mean()) if in_deg.size else 0.0

        if self.stdp and self.stdp.enabled:
            self._build_in_neighbors()

    def compute_global_scalar_threshold(
        self, use_abs_weights: bool = False
    ) -> float:
        """Compute scalar threshold: W̄*N̄_in + 2*I*Tref."""
        weight_matrix = self.synaptic_weights
        if weight_matrix.nnz == 0:
            mean_weight = 0.0
            mean_in_degree = 0.0
        else:
            data = weight_matrix.data
            if use_abs_weights:
                mean_weight = float(np.mean(np.abs(data)))
            else:
                mean_weight = float(np.mean(data))
            # Mean in-degree (mean nnz per column)
            mean_in_degree = float(weight_matrix.getnnz(axis=0).mean())

        two_I_Tref = (
            2.0 * float(self.input_mean_current) * float(self.refractory_period)
        )
        threshold = mean_weight * mean_in_degree + two_I_Tref
        return float(threshold)

    def apply_global_scalar_threshold(self, use_abs_weights: bool = False) -> float:
        """Compute and set global scalar threshold and return it."""
        threshold = self.compute_global_scalar_threshold(use_abs_weights=use_abs_weights)
        self.membrane_threshold = float(threshold)
        return float(threshold)


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
            "mean_in_degree": self.mean_in_degree,
            "time_step": self.time_step,
            "current_amplitude": self.current_amplitude,
        }


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
