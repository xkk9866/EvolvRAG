# EvolvRAG / TSSM

TSSM revises an agent's memory by inferring what changed, who reported it and when the report applied. Its testimony state-space model connects latent facts over time with source reliability and dependencies between reports. The repository contains the model, controlled environments, comparison methods and readers used to study memory revision.

`code/tsm.py` is the current model implementation. The EvolvRAG project name is retained here to keep the earlier retrieval and memory experiments together.

## Code map

| Path | Purpose |
| --- | --- |
| `code/tsm.py` | Latent-state inference and parameter estimation |
| `code/run_tsm.py` | Shared-input evaluation of TSSM and its variants |
| `code/run_experiment.py`, `code/run_fair.py` | Controlled environments and common evaluation inputs |
| `code/dynsim.py`, `code/sim_recovery.py` | Dynamic simulations and recovery experiments |
| `code/run_fusion*.py`, `code/run_td_baselines.py` | Fusion and truth-discovery comparisons |
| `code/build_wikicoach.py`, `code/run_wikicoach.py` | Wikipedia-derived temporal testimony pipeline |
| `code/run_reader_tsm.py`, `code/run_llm_memory.py` | Language-model readers for inferred memory |

## Setup

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
```

The core inference and simulation experiments run independently of a language-model service. Reader experiments additionally need compatible model weights, a GPU and, for the vLLM-based readers, a suitable vLLM installation.

## Run a controlled experiment

From the repository root:

```bash
python code/run_tsm.py 600 stationary
```

The first positional argument is a comma-separated list of seeds, and the second is a comma-separated list of regimes. A third optional argument supplies a JSON dictionary of experiment settings. For example:

```bash
python code/run_tsm.py 600,601 stationary,topic
```

The evaluator uses common document streams for the compared methods and prints per-regime summaries and per-seed scores to standard output. Redirect that output to a file if you want to retain the summary. The main runner uses at most 24 worker processes; its separate `probe` mode accepts `NPROC` to control concurrency. Some experiment drivers run several model fits per world; begin with one seed and one regime when checking a new environment.

`code/variants_*.json` stores variant selections for the corresponding experiment drivers. Inspect the positional arguments in a driver before using a variant file: the scripts represent different experimental protocols and do not share one universal command-line interface.

## Temporal and reader experiments

`build_wikicoach.py` retrieves public Wikidata and Wikipedia records and constructs temporal claims. The downstream scripts read the generated files under `data/`. Other public-data runners access the Hugging Face datasets service. Follow the source dataset's usage terms and keep the downloaded revision and split with each experiment.

Reader programs accept prepared memory windows and local model paths. The Bash schedules use `PYTHON` to select an interpreter and keep model paths under `models/`; adjust the GPU allocation for your machine. Datasets, downloaded content, model weights and generated predictions are not included in this source release.

## Release contents

The source includes current and earlier algorithms, evaluations and diagnostics. Publication text, bibliography tools, figure-generation scripts and submission helpers are excluded. `RELEASE_MANIFEST.json` records the public files and their hashes. The dependency list describes source requirements rather than a frozen training environment.
