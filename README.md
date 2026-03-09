# P&T Treuno 100B

This repository contains the data pipeline and model architecture to natively pre-train **P&T Treuno 100B**. The model is exclusively trained for programming across ALL coding languages, but is configured to heavily oversample and naturally prioritize Elixir.

## Architecture 
The model implements a highly scaled Custom Transformer (10,240 hidden dimension, 72 layers, 80 heads) designed strictly for ~100 Billion Parameters. It utilizes Grouped Query Attention (GQA) and SwiGLU activations.

## 1. Automated Dataset Pipeline
We have built an automated data scraper to pull continuous Elixir data for the model. 

### Setup
1. `pip install requests schedule beautifulsoup4 transformers datasets`
2. Export your GitHub token to avoid strict rate limits:
   ```bash
   export GITHUB_TOKEN="your_personal_access_token"
   ```

### 1. Elixir Data Spiders
Run the scrapers to continuously pull pure Elixir projects and documentation from GitHub and HexDocs:
```bash
python scripts/schedule_auto_updates.py
```

### 2. Multi-Language Base Data 
To teach the model general programming intelligence, run the general code downloader. This script streams billions of multi-language coding tokens (Python, JS, C++, Rust, Go, etc.) from `starcoderdata` into your local dataset without crashing your drive:
```bash
python scripts/download_general_code.py
```

### 3. Cleaning & Merging the Dataset
Before training, process the raw data. This pipeline automatically merges the general programming data with your Elixir data, applying a **10x Oversampling Multiplier** to the Elixir data so the model natively prioritizes it as requested:
```bash
python scripts/data_pipeline.py
```

## 2. Distributed Training a 100B Model
Training a 100B parameter model requires significantly more hardware than consumer GPUs. You must run this across a massive multi-node architecture using DeepSpeed ZeRO-Stage 3, which mathematically shards the optimizer states, gradients, and model parameters across GPUs to fit in memory.

### Prerequisites (Compute Cluster)
- A cluster with e.g. 64 - 1024 GPUs (Nvidia A100 80GB or H100s).
- NVLink / InfiniBand connected nodes. 
- DeepSpeed installed on all nodes (`pip install deepspeed`).

### Launching the Pre-Training Job
On your head node, you must launch DeepSpeed using the `deepspeed` CLI, distributing the `/train.py` script across your hostfile.

```bash
deepspeed --num_gpus=8 --num_nodes=8 --hostfile hostfile.txt train.py
```

**Note**: To dry-run and verify the architecture config without starting the massive training loop:
```bash
python train.py
```
*(Warning: Even trying to allocate 100B parameters randomly in RAM on a single consumer machine will likely OOM kill your process. A minimum of ~200GB of CPU/GPU memory is needed just to hold the FP16 weights).*
