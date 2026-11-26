# LaDIS

**LaDIS (Latent Diffusion for Intrusion Synthesis)** is a Python library for generating synthetic tabular network-traffic observations with latent diffusion.

LaDIS learns a mixed-type representation of numerical and categorical variables with an autoencoder, standardizes the learned continuous latent space, fits a Gaussian diffusion model in that latent domain, and reconstructs generated samples back into a validated tabular representation.

The library is designed to operate directly on **tabular data**. PCAP parsing is optional and is provided only as an auxiliary workflow for users who want to reproduce the flow-level feature-extraction strategy adopted in the associated IoT intrusion synthesis study.

> **LaDIS does not require PCAP files.** If you already have a CSV, Excel file, Parquet file, or pandas `DataFrame`, you can train LaDIS directly from that table.

---

## Contents

- [Overview](#overview)
- [Key features](#key-features)
- [Method overview](#method-overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Input data requirements](#input-data-requirements)
- [Dataset preparation](#dataset-preparation)
- [YAML dataset configuration](#yaml-dataset-configuration)
- [Model configuration](#model-configuration)
- [Model selection](#model-selection)
- [Synthetic generation](#synthetic-generation)
- [Saving and loading models](#saving-and-loading-models)
- [Optional PCAP feature extraction](#optional-pcap-feature-extraction)
- [Extracted network-flow variables](#extracted-network-flow-variables)
- [Bot-IoT Keylogging tutorial](#bot-iot-keylogging-tutorial)
- [Repository structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)

---

## Overview

LaDIS is intended for **mixed-type tabular synthesis**, with a particular focus on network intrusion data containing continuous measurements, discrete counts, binary variables, categorical fields, and high-cardinality network ports.

At a high level, the library performs the following workflow:

```text
tabular data
    │
    ▼
dataset preparation
    │
    ├── schema standardization
    ├── data-quality audit
    ├── feature-role assignment
    ├── deterministic-feature detection
    ├── variable-type inference
    ├── train / validation / test split
    └── optional network-port tokenization
    │
    ▼
mixed-type model representation
    │
    ├── reversible numerical preprocessing
    └── categorical encoding
    │
    ▼
deterministic mixed-type autoencoder
    │
    ▼
continuous latent representation
    │
    ▼
coordinate-wise latent standardization
    │
    ▼
Gaussian latent diffusion
    │
    ▼
synthetic standardized latent vectors
    │
    ▼
inverse latent standardization
    │
    ▼
autoencoder decoder
    │
    ▼
inverse preprocessing
    │
    ▼
deterministic reconstruction
    │
    ▼
validated synthetic tabular data
```

LaDIS is **attack-specific by design** in the intrusion synthesis setting: an independent model is fitted to each traffic type to be generated. Within one fitted dataset, generation is unconditional.

The high-level `LaDIS` API composes preparation, model selection, training, generation, reconstruction, validation, and checkpoint persistence into a single estimator.

---

## Key features

LaDIS provides:

- Mixed numerical and categorical tabular modeling.
- Automatic but auditable variable-type inference.
- Explicit dataset semantics through an optional YAML preparation file.
- Reproducible train/validation/test splitting.
- Train-fitted preprocessing only.
- Optional high-cardinality network-port tokenization.
- Deterministic-feature removal and post-generation reconstruction.
- Reversible numerical preprocessing.
- Count and repeated-mode dequantization before Gaussian quantile transformation.
- Categorical embeddings inside a deterministic mixed-type autoencoder.
- Automatic latent-dimensionality selection.
- Coordinate-wise latent standardization without whitening or rotation.
- Gaussian diffusion in the learned latent space.
- Direct clean-`z0` denoising.
- Controlled denoiser selection.
- Categorical sampling from decoder softmax distributions.
- Post-generation structural and dtype validation.
- Portable model checkpoints.
- Optional PCAP-to-tabular extraction for Bot-IoT and CICIoT2023-style workflows.
- Lightweight integration and regression tests.
- A complete Bot-IoT Keylogging usage notebook.

---

## Method overview

### 1. Dataset preparation

LaDIS begins from a tabular dataset. Preparation is dataset-independent; dataset-specific semantic decisions can be supplied in a YAML file.

The preparation pipeline:

1. Loads the tabular source.
2. Standardizes column names and attempts safe numerical coercion.
3. Audits missing values, infinities, constants, quasi-constants, high-cardinality text, and possible identifiers.
4. Assigns feature roles.
5. Removes observation-specific identifiers and separately stores reconstructable metadata when configured.
6. Validates and removes deterministic derived features when reconstruction rules are available.
7. Infers variable types.
8. Constructs reproducible train, validation, and test partitions.
9. Optionally tokenizes network ports using statistics fitted on the training partition.

No transformation is fitted on validation or test data.

### 2. Numerical representation

Numerical variables are first divided into continuous and discrete-count variables.

Strongly right-skewed, non-negative numerical variables can receive:

```text
log1p → standardization
```

Other numerical variables are standardized directly.

Discrete counts receive local dequantization before quantile transformation. Strongly repeated numerical modes can also be locally dequantized. These operations remove exact ties while preserving an inverse mapping for generation.

A Gaussian quantile transformer is then fitted **only on the training partition**.

The final numerical model representation is continuous and suitable for the autoencoder and latent diffusion pipeline.

### 3. Categorical representation

Categorical variables are encoded as zero-based integer states.

Numerically stored variables can be explicitly modeled as categorical through the dataset configuration. Network ports are the main example.

Categorical states used by the generator are restricted to values observed in the training partition.

### 4. Mixed-type autoencoder

LaDIS uses a deterministic mixed-type autoencoder.

Each categorical variable is projected through its own trainable embedding. The numerical variables and categorical embeddings are concatenated and encoded into a continuous latent vector.

The decoder reconstructs:

- Numerical variables directly.
- One logit vector per categorical variable.

The autoencoder objective combines numerical reconstruction loss with categorical cross-entropy. Categorical losses are normalized by the logarithm of the corresponding category cardinality.

The best validation reconstruction checkpoint is retained and the fitted autoencoder is subsequently frozen.

### 5. Latent standardization

The raw autoencoder latent representation is standardized coordinate by coordinate using statistics fitted on the **training latent vectors only**:

```text
z0 = (r - mean_train) / std_train
```

LaDIS does not apply latent whitening, rotation, clipping, or an additional Gaussianization step.

### 6. Latent diffusion

A Gaussian diffusion model is fitted in the standardized continuous latent space.

The reference LaDIS configuration uses:

- A cosine noise schedule.
- 500 diffusion timesteps.
- Timestep-conditioned MLP denoising.
- Direct prediction of the clean latent vector `z0`.
- Unbounded clean-`z0` predictions.

Training minimizes mean squared error between the predicted and true clean standardized latent vectors.

### 7. Synthetic generation

Generation starts by sampling synthetic standardized latent vectors with the learned reverse diffusion process.

The generated vectors are then:

```text
inverse latent standardization
→ frozen autoencoder decoder
→ inverse Gaussian-quantile transformation
→ repeated-mode restoration
→ inverse numerical preprocessing
→ categorical decoding
→ network-port reconstruction, when configured
→ deterministic feature reconstruction
→ metadata reconstruction, when defined
→ validation
```

Categorical variables are **sampled from the decoder softmax distributions** rather than assigned by `argmax`.

Observation-specific identifiers that were excluded during preparation are not fabricated.

---

## Installation

LaDIS requires Python 3.10 or newer.

From the repository root:

```bash
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

### Optional PCAP dependencies

PCAP parsing is not required for the LaDIS generative pipeline.

Install the optional packet-processing dependencies only if you want to use `ladis.feature_extraction`:

```bash
pip install -e ".[pcap]"
```

The PCAP utilities use Scapy for packet parsing and `tqdm` for progress reporting.

### GPU support

LaDIS uses PyTorch. If `device=None`, the high-level API automatically selects CUDA when it is available and otherwise falls back to CPU.

For non-trivial experiments, a CUDA-capable GPU is strongly recommended.

---

## Quick start

The most direct workflow starts from an existing tabular dataset.

```python
from ladis.pipeline import LaDIS

model = LaDIS(preparation_config_path="config.yaml")

model.fit("data.csv")

synthetic = model.sample(10_000, random_state=42)
```

Save the trained generator:

```python
model.save("ladis_model.joblib")
```

Reload it later without retraining:

```python
from ladis.pipeline import LaDIS

model = LaDIS.load("ladis_model.joblib", device=None, random_state=42)

synthetic = model.sample(10_000, random_state=123)
```

A pandas `DataFrame` can be supplied directly:

```python
model.fit(dataframe)
```

For a one-call fit-and-generate workflow:

```python
synthetic = model.fit_sample(dataframe, num_rows=10_000, random_state=42)
```

---

## Input data requirements

### Supported input types

The preparation pipeline accepts:

- A pandas `DataFrame`.
- CSV.
- Excel.
- Parquet.

A LaDIS dataset is expected to have:

```text
rows = observations
columns = variables
```

For network traffic synthesis, one row normally represents one flow-level observation, but the generative library itself operates on the tabular representation and does not require raw packets.

### Variable types

LaDIS automatically classifies retained variables as:

- `continuous`.
- `binary`.
- `discrete`.
- `categorical`.
- `other`.

The automatic rules are:

1. String/object/pandas categorical variables → categorical.
2. Boolean variables → binary.
3. Numerical variables with exactly two observed values → binary.
4. Remaining integer-valued numerical variables → discrete.
5. Remaining numerical variables → continuous.
6. Empty, constant, datetime, timedelta, or unsupported variables → other.

Manual type overrides can be supplied in the YAML configuration.

Numerically stored variables can also be modeled categorically through `numeric_categorical_columns`.

### Missing and non-finite values

LaDIS audits missing and infinite values during preparation. The final model representation must contain finite numerical values and valid categorical states.

If unsupported variables remain after preparation, fitting fails explicitly rather than silently coercing them into an inappropriate representation.

### Identifiers and provenance

Observation-specific identifiers should not normally be modeled generatively.

Examples include:

- Row IDs.
- Timestamps that uniquely identify observations.
- IP addresses.
- UUIDs.
- Capture-specific identifiers.

The YAML configuration can distinguish between:

- Identifiers that must be excluded.
- Provenance metadata.
- Constant metadata that can be restored after generation.

### Deterministic variables

Variables that can be reconstructed exactly or near-exactly from retained variables can be removed before model fitting and rebuilt after generation.

LaDIS supports configured sum and rate relationships and automatic validation of `_std` / `_var` pairs.

---

## Dataset preparation

The high-level `LaDIS.fit()` method calls the preparation pipeline automatically.

The preparation stages are:

```text
loading
→ schema
→ audit
→ features
→ derived features
→ variable types
→ splitting
→ optional port tokenization
→ model-ready data
```

The same preparation pipeline can also be called directly:

```python
from ladis.preparation import prepare_dataset

result = prepare_dataset("data.csv", config_path="config.yaml")

train = result["data_splits"]["train"]
validation = result["data_splits"]["validation"]
test = result["data_splits"]["test"]

artifacts = result["artifacts"]
reports = result["reports"]
metadata = result["metadata"]
```

The returned object contains:

- `data_splits`.
- `artifacts`.
- `reports`.
- `metadata`.
- `manifest`.
- `bundle_path`.

Preparation results can optionally be persisted as a reusable bundle through the lower-level preparation API.

---

## YAML dataset configuration

The YAML file describes **dataset-specific preparation semantics**. It does **not** define the LaDIS neural architecture.

This distinction is important:

```text
preparation_config_path
    → tells LaDIS what the columns mean and how the dataset should be prepared

model_config
    → controls LaDIS model architecture, optimization, diffusion, and selection
```

### Is a YAML file required?

No.

You can instantiate:

```python
model = LaDIS()
```

and the package will use conservative preparation defaults.

However, a YAML file is recommended whenever the dataset contains semantic information that should not be inferred silently, such as:

- Known identifiers.
- Provenance fields.
- Reconstructable constant metadata.
- Variables that are numerically stored but semantically categorical.
- Network ports.
- Known deterministic sum/rate relationships.
- Dataset-specific split settings.

User-provided YAML values are recursively merged with the package defaults, so a configuration file does not need to redefine every available option.

### Main YAML sections

| Section | Purpose |
|---|---|
| `dataset` | Human-readable dataset metadata |
| `input` | Input parsing and numerical inference |
| `audit` | Thresholds used during structural/data-quality auditing |
| `roles` | Recoverable constants, provenance, and identifiers |
| `variable_types` | Type inference controls and semantic overrides |
| `categorical_analysis` | Rare-category diagnostics |
| `derived_features` | Deterministic sum, rate, and std/variance relationships |
| `split` | Train/validation/test construction |
| `port_tokenization` | Optional network-port treatment |
| `model_preprocessing` | Numerical preprocessing decisions |
| `storage` | Preparation-bundle serialization |

### Generic YAML example

A compact configuration may look like:

```yaml
dataset:
  name: ExampleTraffic
  source_database: ExampleDataset

input:
  numeric_inference_threshold: 1.0
  sheet_name: 0

roles:
  recoverable_constant_columns:
    - label

  provenance_columns: []

  expected_identifier_candidates:
    - flow_id
    - src_ip
    - dst_ip

  identifier_columns_to_exclude:
    - flow_id
    - src_ip
    - dst_ip

  remove_approximate_constant_columns: true

variable_types:
  integer_tolerance: 1.0e-9
  overrides: {}
  numeric_categorical_columns:
    - src_port
    - dst_port

derived_features:
  sum_rules:
    total_bytes:
      - fwd_bytes
      - bwd_bytes

  rate_rules:
    flow_bytes_s:
      numerator: total_bytes
      denominator: flow_duration
      scale: 1.0

  std_var_detection:
    enabled: true
    std_suffix: "_std"
    var_suffix: "_var"
    rtol: 1.0e-5
    atol: 1.0e-8
    minimum_match_pct: 99.9
    minimum_valid_coverage_pct: 99.9

split:
  train_size: 0.80
  validation_size: 0.10
  test_size: 0.10
  random_state: 42
  keep_exact_duplicates_together: true
  maximum_acceptable_deviation_pct: 0.5

port_tokenization:
  columns:
    - src_port
    - dst_port
  top_k_ports: 50
  minimum_exact_count: 20

model_preprocessing:
  skewness_log_threshold: 2.0
```

### `dataset`

```yaml
dataset:
  name: ExampleTraffic
  source_database: ExampleDataset
```

These fields are metadata and do not alter model architecture.

### `input`

```yaml
input:
  numeric_inference_threshold: 1.0
  sheet_name: 0
```

`numeric_inference_threshold` controls safe conversion of string-like columns to numeric storage.

`sheet_name` is used for Excel inputs.

### `audit`

The audit configuration controls diagnostic thresholds for:

- Quasi-constant variables.
- High-cardinality text.
- Almost-unique variables.
- Approximate numerical constants.

The audit itself is diagnostic: a variable is not removed merely because it is flagged as identifier-like.

### `roles`

`roles` contains semantic decisions that should be explicit.

#### Recoverable constants

```yaml
recoverable_constant_columns:
  - label
```

These columns are removed from the model representation and restored after generation when their value is well defined.

#### Provenance

```yaml
provenance_columns:
  - pcap
```

Provenance variables are stored separately from generative features.

Whether they are included in the final synthetic table can be controlled through `include_provenance`.

#### Identifier candidates

```yaml
expected_identifier_candidates:
  - flow_start
  - flow_end
  - src_ip
  - dst_ip
```

This documents columns expected to behave like identifiers.

#### Identifier exclusion

```yaml
identifier_columns_to_exclude:
  - flow_start
  - flow_end
  - src_ip
  - dst_ip
```

These variables are excluded from generative modeling and are not fabricated after generation.

### `variable_types`

LaDIS can override automatically inferred variable types:

```yaml
variable_types:
  overrides:
    some_column: categorical
```

Accepted override types are:

```text
continuous
binary
discrete
categorical
other
```

Numerically stored variables that should be modeled categorically can be declared separately:

```yaml
numeric_categorical_columns:
  - src_port
  - dst_port
```

### `derived_features`

Configured deterministic relationships can be removed before model fitting and reconstructed after generation.

#### Sum rules

```yaml
sum_rules:
  total_bytes:
    - fwd_bytes
    - bwd_bytes
```

represents:

```text
total_bytes = fwd_bytes + bwd_bytes
```

#### Rate rules

```yaml
rate_rules:
  flow_bytes_s:
    numerator: total_bytes
    denominator: flow_duration
    scale: 1.0
```

represents:

```text
flow_bytes_s = total_bytes / flow_duration
```

#### Standard deviation / variance detection

When enabled, LaDIS can validate relationships such as:

```text
x_var ≈ x_std²
```

using configured suffixes and tolerances.

### `split`

The default split is approximately:

```text
80% training
10% validation
10% test
```

The split is reproducible through `random_state`.

Exact duplicate observations can be kept together. The current preparation pipeline can also ensure that categorical states required by the model remain represented in the training partition.

The test partition is not used for model fitting or model selection.

### `port_tokenization`

Network ports can be modeled categorically without creating one state for every observed integer port.

The tokenizer is fitted on the training partition only.

Frequently observed ports can be retained as exact tokens, while remaining ports are assigned to semantic ranges. The empirical training distribution associated with each token is stored so that generated tokens can later be converted back into concrete port numbers.

### `model_preprocessing`

Currently this section controls the skewness threshold used to decide whether a non-negative numerical variable receives `log1p` before standardization.

### `storage`

Storage options define preparation-bundle serialization, including Parquet and joblib compression.

### Bot-IoT Keylogging configuration

A complete real configuration is provided at:

```text
configs/bot_iot_keylogging.yaml
```

It documents the semantic roles, deterministic relationships, port treatment, split settings, and reconstruction rules used for the Bot-IoT Keylogging tutorial.

---

## Model configuration

The YAML file and the model configuration serve different purposes.

```python
model = LaDIS(preparation_config_path="config.yaml", model_config=None)
```

### Default model configuration

When:

```python
model_config=None
```

LaDIS uses its complete reference model configuration.

Key defaults include:

| Component | Default |
|---|---|
| Latent dimension candidates | `(4, 8, 12, 16, 20, 24, 28, 32)` |
| Categorical embedding dimension | bounded between 2 and 16 |
| AE encoder | `(256, 128, 64)` |
| AE decoder | `(64, 128, 256)` |
| AE learning rate | `1e-3` |
| AE weight decay | `1e-5` |
| AE batch size | `4096` |
| AE maximum epochs | `100` |
| AE early-stopping patience | `10` |
| Diffusion hidden dimensions | `(256, 512, 512, 256)` |
| Timestep embedding dimension | `128` |
| Diffusion timesteps | `500` |
| Noise schedule | cosine |
| Prediction target | clean `z0` |
| Diffusion learning rate | `5e-4` |
| Diffusion weight decay | `1e-5` |
| Diffusion batch size | `4096` |
| Diffusion training steps | `10000` |
| Candidate evaluation ceiling | `10000` validation observations |

For datasets with fewer than 10,000 validation observations, candidate evaluation uses the available validation-set size.

### Custom model configuration

A user configuration is recursively merged with the reference configuration.

For example:

```python
from ladis.pipeline import LaDIS

model = LaDIS(preparation_config_path="config.yaml", model_config={"latent_dimension_candidates": (8, 16, 24), "autoencoder": {"maximum_epochs": 50}})
```

Only the supplied entries are replaced.

The reference pipeline requires:

```text
prediction_target = "z0"
bound_predicted_z0 = False
```

because clean-`z0` prediction is part of the LaDIS formulation.

### Reduced smoke-test configuration

For integration testing or demonstrations:

```python
from ladis.pipeline import build_ladis_smoke_test_config

model_config = build_ladis_smoke_test_config()
```

This drastically reduces training budgets while preserving the complete software workflow.

It is **not intended for scientific experiments or model-quality conclusions**.

---

## Model selection

LaDIS performs model selection using training and validation data only.

The test partition is excluded from all selection decisions.

### Stage 1: latent dimension

For each candidate latent dimension:

1. A complete mixed-type autoencoder is trained.
2. The autoencoder is frozen.
3. The training latent representation is standardized.
4. A reference clean-`z0` diffusion model is trained.
5. Synthetic validation observations are generated.
6. Decoded synthetic data are evaluated with SDMetrics.

The number of evaluation observations is:

```text
min(configured evaluation ceiling, available validation observations)
```

The default ceiling is 10,000.

Each candidate is evaluated against the same fixed validation reference.

The following metrics are independently ranked in descending order:

- Overall SDMetrics Quality Score.
- Column Shapes.
- Column Pair Trends.

The candidate with the lowest mean rank is selected. Overall Quality Score is the deterministic tie-breaker.

The autoencoder is then retrained at the selected dimensionality.

### Stage 2: latent denoiser

With the selected latent representation fixed, LaDIS evaluates four clean-`z0` denoiser configurations:

| Candidate | Hidden dimensions | Learning rate |
|---|---:|---:|
| reference | `(256, 512, 512, 256)` | `5e-4` |
| lower LR | `(256, 512, 512, 256)` | `2.5e-4` |
| wide | `(512, 1024, 1024, 512)` | `5e-4` |
| wide + lower LR | `(512, 1024, 1024, 512)` | `2.5e-4` |

Selection follows this deterministic hierarchy:

1. Higher synthetic Quality Score.
2. Lower mean absolute latent correlation error.
3. Smaller deviation of the median generated-to-training latent standard-deviation ratio from 1.
4. Fewer model parameters.

The already trained winning denoiser checkpoint is retained directly. It is not refitted after selection.

### Inspecting selection results

After fitting:

```python
print(model.selected_latent_dimension_)
print(model.selected_diffusion_candidate_)
```

Detailed results are available through:

```python
model.latent_dimension_selection_["study"]
model.latent_dimension_selection_["ranking"]

model.denoiser_selection_["comparison"]
model.denoiser_selection_["selection"]
```

---

## Synthetic generation

Generate an arbitrary number of observations from a fitted model:

```python
synthetic = model.sample(50_000, random_state=42)
```

### Validation

Generation-time validation is enabled by default:

```python
synthetic = model.sample(50_000, validate=True)
```

The high-level reconstruction validation checks the available schema and domain constraints, including:

- Reconstructed column order.
- dtypes.
- Missing values.
- Infinite values.
- Constant labels when configured.
- Protocol support when available.
- Valid port values.
- Binary variable support.
- Deterministic derived-feature relationships.

### Diagnostics

Return generation diagnostics together with the synthetic table:

```python
synthetic, diagnostics = model.sample(50_000, random_state=42, return_diagnostics=True)
```

Diagnostics include:

- Latent sampling summary.
- Decoder summary.
- Reconstruction corrections.
- Reconstruction validation summary.
- Derived-feature validation.
- dtype validation.

The most recent diagnostics are also stored as:

```python
model.last_sample_diagnostics_
```

### Provenance

The estimator accepts:

```python
include_provenance=True
```

at construction time, and `sample()` can override it.

For newly generated network traffic, it is often more appropriate to use:

```python
include_provenance=False
```

because synthetic observations do not originate from a real capture file.

Observation-specific identifiers excluded during preparation are not regenerated.

---

## Saving and loading models

A fitted LaDIS generator can be saved as a portable joblib checkpoint:

```python
model.save("ladis_model.joblib")
```

Compression can be changed:

```python
model.save("ladis_model.joblib", compress=3)
```

The checkpoint stores the trained autoencoder, selected diffusion model, preprocessing state, latent standardization, categorical mappings, port artifacts, and reconstruction metadata required for later generation.

Reload a model with:

```python
from ladis.pipeline import LaDIS

loaded_model = LaDIS.load("ladis_model.joblib", device=None, random_state=42, verbose=False)
```

Then generate without fitting again:

```python
synthetic = loaded_model.sample(10_000, random_state=123)
```

---

## Optional PCAP feature extraction

> **PCAP feature extraction is optional.**
>
> LaDIS itself is a tabular generative library. The `ladis.feature_extraction` package is provided to reproduce the network-flow construction strategy adopted in the associated study and to offer a convenient PCAP-to-table path for compatible datasets.

The relationship is:

```text
PCAP files
    │
    │ optional
    ▼
ladis.feature_extraction
    │
    ▼
flow-level tabular data
    │
    ▼
LaDIS
```

If a feature table already exists, skip this section entirely and call `LaDIS.fit()` on the table.

Install the PCAP dependencies with:

```bash
pip install -e ".[pcap]"
```

Supported capture extensions are:

```text
.pcap
.pcapng
.cap
```

### Shared flow extraction

The common extractor is:

```python
from ladis.feature_extraction import PCAPFeatureExtractor

extractor = PCAPFeatureExtractor(flow_timeout=30.0, max_packets_per_flow=5000, show_progress=True)
```

The experimental defaults are:

```text
flow inactivity timeout = 30 seconds
packet-count threshold  = 5000
```

Flows are bidirectional and defined by a canonical endpoint ordering over:

```text
(source IP, source port)
(destination IP, destination port)
protocol
```

The resulting "forward" direction is therefore the canonical direction of the flow representation; it should not be interpreted as a universal client→server or request→response direction.

### Bot-IoT

Bot-IoT requires ground-truth matching because one scenario capture can contain both attack and benign flows.

Example:

```python
from ladis.feature_extraction import PCAPFeatureExtractor, discover_pcap_files, extract_botiot_pcaps

pcap_files = discover_pcap_files("/path/to/pcaps")

extractor = PCAPFeatureExtractor(flow_timeout=30.0, max_packets_per_flow=5000)

result = extract_botiot_pcaps(pcap_files, ground_truth_path="/path/to/ground_truth.csv", scenario_label="Keylogging", extractor=extractor,
    output_path=None, verbose=True)

data = result.data
diagnostics = result.diagnostics
```

Bot-IoT matching uses:

- The canonical bidirectional flow key.
- Normalized protocol and port values.
- Temporal overlap between extracted flows and ground-truth intervals.

When more than one ground-truth row overlaps a flow, the row with the greatest overlap duration is selected.

Unmatched flows are excluded from the final matched table.

Normal traffic remains labeled `Normal`; matched attack traffic receives the supplied scenario label.

The Keylogging tutorial additionally separates the matched traffic into attack-specific and benign tables before fitting LaDIS.

### CICIoT2023

For CICIoT2023-style scenario folders, one label is assigned to all flows extracted from the selected PCAP set.

```python
from ladis.feature_extraction import extract_ciciot_folder

result = extract_ciciot_folder("/path/to/scenario_folder", label="Recon-OSScan")

data = result.data
diagnostics = result.diagnostics
```

No Bot-IoT-style ground-truth flow matching is performed in this workflow.

---

## Extracted network-flow variables

The optional PCAP extractor produces a **114-column extracted flow table**.

This should not be described as "114 model features": the table also contains provenance, observation-specific metadata, and the traffic label, and the later preparation stage decides which columns are modeled.

The canonical table contains:

```text
23 base / flow-level columns
66 statistical descriptor columns
24 TCP flag counter columns
 1 traffic label
-------------------------------
114 total columns
```

### Base flow columns

| Column | Meaning |
|---|---|
| `pcap` | Source capture filename |
| `flow_start` | First packet timestamp in the extracted flow |
| `flow_end` | Last packet timestamp in the extracted flow |
| `src_ip` | Canonical forward source IP |
| `dst_ip` | Canonical forward destination IP |
| `src_port` | Canonical forward source port |
| `dst_port` | Canonical forward destination port |
| `protocol` | Transport/network protocol representation |
| `flow_duration` | Flow duration with a lower numerical bound of `1e-9` |
| `total_pkts` | Total packet count |
| `total_bytes` | Total packet bytes |
| `fwd_pkts` | Canonical forward packet count |
| `bwd_pkts` | Canonical backward packet count |
| `fwd_bytes` | Canonical forward packet bytes |
| `bwd_bytes` | Canonical backward packet bytes |
| `flow_pkts_s` | Total packets per second |
| `flow_bytes_s` | Total bytes per second |
| `fwd_pkts_s` | Forward packets per second |
| `bwd_pkts_s` | Backward packets per second |
| `fwd_bytes_s` | Forward bytes per second |
| `bwd_bytes_s` | Backward bytes per second |
| `down_up_ratio` | Backward/forward packet-count ratio, using `max(fwd_pkts, 1)` in the denominator |
| `byte_down_up_ratio` | Backward/forward byte-count ratio, using `max(fwd_bytes, 1)` in the denominator |

### Statistical descriptor families

For each of the following 11 quantities:

```text
pkt_len
fwd_pkt_len
bwd_pkt_len
payload_len
fwd_payload_len
bwd_payload_len
header_len
ttl
flow_iat
fwd_iat
bwd_iat
```

the extractor computes:

```text
_min
_max
_mean
_std
_var
_sum
```

This produces:

```text
11 × 6 = 66 statistical columns
```

Examples include:

```text
pkt_len_mean
payload_len_std
ttl_max
flow_iat_sum
fwd_iat_var
bwd_pkt_len_min
```

The standard deviation and variance use NumPy population definitions (`ddof=0`).

If a packet-derived vector is empty, all six descriptors are set to zero.

### TCP flag counters

The extractor tracks eight TCP flags:

```text
FIN
SYN
RST
PSH
ACK
URG
ECE
CWR
```

For each flag it stores:

```text
<flag>_cnt
fwd_<flag>_cnt
bwd_<flag>_cnt
```

This produces:

```text
8 × 3 = 24 flag columns
```

For historical schema compatibility, the CWR flag uses the stem:

```text
cwe
```

so the corresponding columns are:

```text
cwe_cnt
fwd_cwe_cnt
bwd_cwe_cnt
```

although they represent TCP **CWR**.

### Label

The last column is:

```text
label
```

Its assignment depends on the dataset-specific extraction workflow.

### Extraction semantics

The extractor follows these definitions:

- Non-IPv4 packets are ignored.
- TCP and UDP use their observed ports.
- ICMP and other supported non-TCP/UDP IP traffic use port `0`.
- Payload length is packet length minus IP and TCP/UDP headers, clipped at zero.
- TTL is taken from the IPv4 header.
- Inter-arrival times are computed after sorting timestamps and applying `np.diff`.
- Inactive flows are closed after a 30 s gap by default.
- The packet-count threshold reproduces the experimental implementation exactly.

The feature-extraction code is intentionally separate from the core LaDIS generative pipeline.

---

## Bot-IoT Keylogging tutorial

The repository includes:

```text
examples/01_botiot_keylogging.ipynb
```

This notebook is a complete worked example of the library.

It demonstrates:

```text
Bot-IoT PCAPs + ground truth
    │
    ▼
optional feature extraction
    │
    ▼
Bot-IoT ground-truth matching
    │
    ▼
Keylogging / Normal separation
    │
    ▼
Keylogging preparation YAML
    │
    ▼
LaDIS.fit()
    │
    ▼
latent-dimension selection
    │
    ▼
denoiser selection
    │
    ▼
LaDIS.sample()
    │
    ▼
synthetic Keylogging table
    │
    ├── save CSV
    └── save checkpoint
            │
            ▼
        reload model
            │
            ▼
        generate more data
```

### Dataset availability

The repository does not redistribute Bot-IoT PCAPs or ground-truth data.

Download the original dataset separately and configure the notebook paths locally.

### Full protocol vs reduced demonstration

The notebook contains:

```python
RUN_FULL_PAPER_PROTOCOL = False
```

When `False`, it uses:

```python
build_ladis_smoke_test_config()
```

This reduced configuration is intended only as a fast software demonstration.

Set:

```python
RUN_FULL_PAPER_PROTOCOL = True
```

to use the complete default LaDIS model configuration.

### Keylogging YAML

The notebook uses:

```text
configs/bot_iot_keylogging.yaml
```

This file describes the dataset-specific preparation semantics for the extracted Keylogging table.

It does not configure the neural architecture.

### Starting from an already extracted table

The PCAP cells are optional.

If `Keylogging_features.csv` already exists:

```python
from ladis.pipeline import LaDIS

model = LaDIS(preparation_config_path="configs/bot_iot_keylogging.yaml")

model.fit("/path/to/Keylogging_features.csv")

synthetic = model.sample(10_000, include_provenance=False)
```

The same principle applies to a different dataset: start from a compatible tabular source and provide an appropriate dataset-specific YAML file when semantic preparation decisions are required.

---

## Repository structure

```text
LaDIS/
├── ladis/
│   ├── preparation/
│   │   ├── audit.py
│   │   ├── config.py
│   │   ├── derived_features.py
│   │   ├── features.py
│   │   ├── loading.py
│   │   ├── pipeline.py
│   │   ├── ports.py
│   │   ├── schema.py
│   │   ├── splitting.py
│   │   ├── storage.py
│   │   └── types.py
│   │
│   ├── feature_extraction/
│   │   ├── core.py
│   │   ├── botiot.py
│   │   └── ciciot.py
│   │
│   ├── preprocessing.py
│   ├── models.py
│   ├── training.py
│   ├── latent.py
│   ├── sampling.py
│   ├── selection.py
│   ├── reconstruction.py
│   ├── evaluation.py
│   ├── io.py
│   ├── pipeline.py
│   └── utils.py
│
├── configs/
│   └── bot_iot_keylogging.yaml
│
├── examples/
│   └── 01_botiot_keylogging.ipynb
│
├── tests/
│   ├── test_feature_extraction.py
│   ├── test_io.py
│   ├── test_pipeline.py
│   └── test_selection.py
│
└── pyproject.toml
```

### Main modules

| Module | Responsibility |
|---|---|
| `ladis.pipeline` | High-level `LaDIS` estimator |
| `ladis.preparation` | Dataset loading, auditing, semantics, splitting, and port preparation |
| `ladis.preprocessing` | Mixed-type model preprocessing and inverse transformations |
| `ladis.models` | Autoencoder and latent diffusion model definitions |
| `ladis.training` | Autoencoder and diffusion training |
| `ladis.latent` | Latent encoding, freezing, and standardization |
| `ladis.sampling` | Reverse diffusion and decoder sampling |
| `ladis.selection` | Latent-dimension and denoiser selection |
| `ladis.reconstruction` | Reconstruction of original-space constraints and metadata |
| `ladis.evaluation` | Model-space and synthetic-data diagnostics |
| `ladis.io` | Portable checkpoint serialization and restoration |
| `ladis.feature_extraction` | Optional PCAP-to-tabular feature extraction |

---

## Reproducibility

LaDIS is designed so that dataset semantics and fitted preprocessing state are explicit and reusable.

### Train-only fitting

The following are fitted from the training partition rather than the full dataset:

- Numerical transformations.
- Categorical mappings.
- Network-port tokenization.
- Count dequantization intervals.
- Repeated-mode dequantization intervals.
- Gaussian quantile transformation.
- Autoencoder parameters.
- Latent standardization.
- Latent diffusion parameters.

Validation data are used for model selection.

Test data are excluded from fitting and selection.

### Random seeds

Random states control:

- Dataset splitting.
- Preprocessing stochasticity.
- Model initialization/training.
- Latent sampling.
- Categorical sampling.
- Network-port reconstruction.

For reproducible experiments, set the dataset split seed in YAML and pass an explicit `random_state` when constructing or sampling from the estimator.

### Checkpoints

`LaDIS.save()` stores the components required to reconstruct a fitted generator later, including preprocessing and reconstruction state.

This allows generation to continue without repeating model fitting or model selection.

### Dataset configuration

Keep the YAML configuration used for each dataset together with experimental outputs.

It records semantic decisions that cannot always be inferred safely from column values alone.

### Software versions

The package includes utilities for recording the principal Python, NumPy, pandas, PyTorch, CUDA, joblib, scikit-learn, and SDMetrics versions used by an experiment.

### Validated scientific environment

The complete LaDIS pipeline used for the reference experiments has been validated in the following environment:

| Component | Validated version |
|---|---|
| Operating system | Linux x86_64, glibc 2.28 |
| GPU | NVIDIA A40 |
| Python | 3.14.4 |
| NumPy | 2.4.4 |
| pandas | 2.3.3 |
| PyTorch | 2.11.0+cu126 |
| CUDA | 12.6 |
| scikit-learn | 1.8.0 |
| SDMetrics | 0.33.0 |
| joblib | 1.5.3 |

These versions describe the environment used to validate the complete scientific LaDIS workflow. Other operating systems, hardware configurations, and compatible dependency versions may also work, but have not yet been systematically validated.

---

## Testing

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the complete test suite:

```bash
pytest -q
```

The tests cover distinct parts of the package, including:

- Optional feature-extraction schema and helpers.
- Checkpoint I/O.
- High-level pipeline integration.
- Latent dimension ranking.
- Controlled denoiser candidate construction.
- Denoiser selection hierarchy.
- Prevention of unnecessary post-selection refitting.

A useful additional syntax check is:

```bash
python -m compileall -q ladis tests
```

---

## Citation

If you use LaDIS in academic work, please cite the associated manuscript:

> E. Sánchez-Carballo, F. M. Melgarejo-Meseguer, and J. L. Rojo-Álvarez,
> "Latent Diffusion for Internet of Things Attack Data Generation in Intrusion Detection,"
> *arXiv preprint arXiv:2601.16976*, 2026.

```bibtex
@misc{sanchezcarballo2026latent,
  title = {Latent Diffusion for Internet of Things Attack Data Generation in Intrusion Detection},
  author = {Sánchez-Carballo, Estela and Melgarejo-Meseguer, Francisco M. and Rojo-Álvarez, José Luis},
  year = {2026},
  eprint = {2601.16976},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

A peer-reviewed journal version is currently under consideration. Once the final journal article is published, please cite the journal version in preference to the arXiv preprint.

---

## License

LaDIS is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 Estela Sánchez-Carballo.

This license applies to the LaDIS source code and repository documentation distributed under these terms. Third-party datasets, external resources, and software dependencies remain subject to their respective licenses and terms of use.

The optional PCAP feature-extraction utilities do not grant any rights to the underlying Bot-IoT, CICIoT2023, or other external datasets. Users are responsible for obtaining and using those datasets in accordance with the terms established by their respective providers.