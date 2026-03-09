# P&T Treuno (Free Tier - 125M)

This repository contains the data pipeline and model architecture to natively pre-train **P&T Treuno**. The model is exclusively trained for programming across ALL coding languages, but is configured to heavily oversample and naturally prioritize Elixir.

## Architecture (Scaled for Free GPUs)
To allow free training on platforms like Google Colab, the architecture has been scaled down to roughly **125 Million Parameters** (similar to GPT-2 Small). It utilizes a custom 12-layer Llama-style CausalLM architecture.

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

## 2. Training the Model (For Free)

Since the architecture is scaled down to ~125M parameters, it will easily fit on the free 16GB GPUs provided by Google Colab or Kaggle.

### How to Train on Google Colab (Free)
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **File > New Notebook**
3. At the top menu, click **Runtime > Change runtime type**
4. Select **T4 GPU** and click Save.
5. In the first code block, clone your repository and install dependencies:
   ```bash
   !git clone https://github.com/phravins/Treuno-100B.git
   %cd Treuno-100B
   !pip install transformers datasets accelerate
   ```
6. *(Optional)* Run the dataset commands above inside Colab to fetch fresh data.
7. Start the training loop!
   ```bash
   !python train.py
   ```

Because we removed the DeepSpeed requirement, it will natively use the PyTorch/HuggingFace Trainer and fit perfectly inside the free 16GB T4 GPU!

_TREUNO-125M is an first version P&T Based model it is on under progress_
