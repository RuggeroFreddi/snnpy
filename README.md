# 🧠 Spiking Neural Network (SNN) – Reservoir Model

Questa classe implementa un modello di **Spiking Neural Network (SNN)** basato su neuroni **LIF** e struttura **reservoir**, con topologia **small-world** generata tramite il modello di Watts-Strogatz. Il flusso operativo tipico del modello è il seguente:

1. **Istanziare** una rete SNN.
2. **Fornire** una matrice binaria `input_spike_times` con shape `(num_input_neurons, time_steps)` dove `1` indica la stimolazione di un neurone di input a un certo tempo.
3. A ogni passo temporale, viene iniettata una corrente **pari alla soglia** nei neuroni di input attivati.

> ⚠️ I neuroni di **input** sono selezionati casualmente tra i primi neuroni e il loro numero è pari al numero di righe di `input_spike_times`.  
> ⚠️ I neuroni di **output** sono selezionati **casualmente tra i neuroni nascosti**, ma è anche possibile specificarli manualmente passando un vettore di indici con `set_output_neurons()`.

---

## ⚙️ Parametri settabili (`SimulationParams`)

| Parametro                | Tipo               | Descrizione                                                                                     | Restrizioni e default                                                        |
|--------------------------|--------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `num_neurons`            | `int`              | Numero totale di neuroni nella rete (input + hidden + output)                                   | ≥ 1. Richiesto solo se `adjacency_matrix` **non** è fornito                  |
| `num_output_neurons`     | `int`              | Numero di neuroni di output selezionati automaticamente dal reservoir                           | ≥ 1. Richiesto solo se `output_neurons` **non** è fornito                    |
| `output_neurons`         | `np.ndarray[int]`  | Indici dei neuroni di output scelti manualmente                                                 | Array 1D di interi in `[0, num_neurons)`. Richiesto solo se `num_output_neurons` **non** è fornito |
| `membrane_threshold`     | `float`            | Soglia di potenziale per generare uno spike                                                     | > 0                                                                           |
| `leak_coefficient`       | `float`            | Fattore di decadimento del potenziale a ogni timestep                                           | Intervallo `[0, 1)`                                                           |
| `refractory_period`      | `int`              | Numero di passi temporali dopo uno spike durante i quali un neurone non può emettere un altro spike                          | ≥ 0                                                                           |
| `duration`               | `int`              | Durata totale della simulazione in timestep                                                     | ≥ 1 se specificato. Default: lunghezza di `input_spike_times`                            |
| `input_spike_times`      | `np.ndarray[int]`  | Matrice binaria (neuroni × tempo) con le stimolazioni in input                                  | Array 2D di 0/1                                                            |
| `membrane_potentials`    | `np.ndarray[float]`| Potenziali iniziali dei neuroni                                                                  | Array 1D con valori in `[0, membrane_threshold]`                              |
| `adjacency_matrix`       | `scipy.sparse`     | Matrice di adiacenza pesata della rete neurale                                                  | Matrice quadrata, pesi float. Richiesto solo se `num_neurons` **non** è fornito |
| `mean_weight`            | `float`            | Peso medio delle connessioni sinaptiche                             | > 0. Richiesto solo se `adjacency_matrix` non è fornito                                |
| `weight_variance`        | `float`            | Varianza dei pesi sinaptici                                         | ≥ 0.  Richiesto solo se `adjacency_matrix` non è fornito. Default: `0.1 * mean_weight`                                            |
| `current_amplitude`      | `float`            | Ampiezza della corrente di input                                                                | Qualsiasi numero reale. Default: `membrane_threshold`                         |
| `small_world_graph_p`    | `float`            | Probabilità di ri-connessione nella topologia small-world (Watts-Strogatz)                      | Intervallo `[0, 1]`. Richiesto solo se `is_random_uniform` è `False`         |
| `small_world_graph_k`    | `int`              | Ogni nodo inizialmente connesso a `k` vicini nella topologia small-world                        | Intero pari ≥ 2. Richiesto solo se `is_random_uniform` è `False`             |
| `connection_prob`        | `float`            | Probabilità di connessione casuale                            | Intervallo `[0, 1]`. Richiesto solo se `is_random_uniform` è `True`          |
| `is_random_uniform`      | `bool`             | Se `True`, genera una rete casuale uniforme; altrimenti usa una topologia small-world           | Default: False. Esclusivo con `small_world_graph_k` e `small_world_graph_p` se `True` |

---

## 🧩 Metodi principali

### ▶️ Simulazione

- `simulate()`  
  Esegue la simulazione per l’intera durata temporale e restituisce un array bidimensionale NumPy (`np.ndarray`):
  
  - matrice binaria di dimensione `[tempo × neuroni_output]`, in cui ogni riga rappresenta un timestep e ogni colonna un neurone di output. L’elemento è `1` se il neurone ha generato uno spike in quell’istante, `0` altrimenti.

- `get_spike_time_lists_output()`  
  Restituisce una lista di liste: ciascuna sottolista contiene gli istanti temporali (timestep) in cui ogni neurone di output ha generato uno spike.

- `set_input_spike_times(input_spike_times)`  
  Imposta la matrice binaria degli spike di input (forma `[neuroni_input (0/1) × tempo]`).  
  Aggiorna automaticamente la durata della simulazione (`duration`) se non già impostata.

- `set_membrane_potentials(membrane_potentials)`  
  Imposta i potenziali iniziali di membrana per ciascun neurone.  
  L'array deve essere 1D con lunghezza uguale al numero di neuroni (`num_neurons`).

- `set_output_neurons(indices)`  
  Imposta manualmente i neuroni di output specificando i loro indici (array di interi).  
  Verifica che gli indici siano validi (`0 ≤ indice < num_neurons`).

- `reset()`  
  Azzera lo stato interno della rete per eseguire una nuova simulazione.  
  Ripristina i potenziali di membrana iniziali, azzera spike matrix, spike count e timer refrattario, mantenendo inalterati tutti gli altri parametri (topologia, neuroni di output, ecc.).
---

### 📈 Feature extraction (dopo la simulazione)
Ogni metodo di estrazione delle feature restituisce un array unidimensionale NumPy (np.ndarray) contenente una feature per ciascun neurone di output.

- `extract_features_from_spikes()`  
  Estrae tutte le principali feature dai neuroni di output (spike count, entropy, isi, etc.)

- `get_spike_counts()`  
  Totale spike per neurone.

- `get_mean_spike_times()`, `get_first_spike_times()`, `get_last_spike_times()`  
  Statistiche temporali sui momenti degli spike.

- `get_mean_isi_per_neuron()`, `get_isi_variance_per_neuron()`  
  Statistiche sugli intervalli tra spike (ISI).

- `get_spike_entropy_per_neuron()`  
  Entropia della distribuzione temporale degli spike.

- `get_spike_rates()`  
  Tasso medio di spike (spike/time).

- `get_autocorrelation_first_lag()`  
  Autocorrelazione al primo lag (indicatore di regolarità).

- `get_burstiness()`  
  Burstiness index (variazione nell’ISI).

- `get_spike_symmetry()`  
  Simmetria spike tra prima e seconda metà temporale.

- `get_spike_histogram_moments()`  
  Media, skewness e curtosi di un istogramma temporale degli spike.

---

### 🧪 Utility

- `calculate_mean_isi()`  
  ISI medio su tutti i neuroni.

- `reset_synaptic_weights(mean, std)`  
  Rigenera tutti i pesi sinaptici esistenti da una distribuzione normale.

---

### 💾 Salvataggio/Caricamento

- `save_topology()`, `load_topology()`  
  Salva/carica la matrice dei pesi sinaptici (`adjacency_matrix`) su disco nel formato `.npz` (matrice sparsa SciPy).

- `save_membrane_potentials()`, `load_membrane_potentials()`  
  Salva/carica i potenziali di membrana in formato `.npy`.

- `save_output_neurons()`, `load_output_neurons()`  
  Salva/carica gli indici dei neuroni di output in formato `.npy`.

- `set_topology(sparse_matrix)`, `get_topology()`  
  Imposta o restituisce la matrice dei pesi sinaptici. *(Nota: `set_topology()` non è esplicito nel codice ma può essere aggiunto facilmente.)*

- `set_membrane_potentials(array)`, `get_membrane_potentials()`  
  Imposta o restituisce i potenziali di membrana.

- `set_output_neurons(indices)`, `get_output_neurons()`  
  Imposta o restituisce gli indici dei neuroni di output.

> ℹ️ **Nota:** I metodi `load_topology()`, `load_output_neurons()` e `load_membrane_potentials()`  
> sono disponibili anche come **funzioni standalone**, utili per caricare dati **prima** di istanziare l’oggetto `SNN`.

---

## 📁 File di output

| Funzione                  | Percorso di default                      |
|--------------------------|------------------------------------------|
| Topologia sinaptica      | `dati/snn_matrices.npz`                  |
| Potenziali di membrana   | `dati/membrane_potentials.npy`           |
| Neuroni di output        | `dati/output_neurons.npy`                |

```python
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

snn = SNN(params)
output = snn.simulate()
features = snn.extract_features_from_spikes()
```