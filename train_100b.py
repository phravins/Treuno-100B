import os
import sys
import logging
from dataclasses import dataclass, field
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION FOR P&T TREUNO 100B ---
# We define an architecture based on Llama scaled up to ~100B parameters
# WARNING: Training a 100B model from scratch requires massive compute resources 
# (e.g., hundreds or thousands of H100/A100 GPUs) and DeepSpeed ZeRO-3.
TREUNO_100B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 10240,            # High dimensionality for 100B
    "intermediate_size": 32768,      # SwiGLU intermediate size
    "num_hidden_layers": 80,         # 80 Layers
    "num_attention_heads": 80,       # 80 Attention heads
    "num_key_value_heads": 8,        # Grouped Query Attention (GQA) for efficiency
    "max_position_embeddings": 8192, # Large Context Window for coding
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
    "vocab_size": 32000              # Standard tokenizer vocab size
}

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "processed_dataset.jsonl")
# Using DeepSeek base tokenizer as placeholder, must match vocab_size in config
TOKENIZER_NAME = "deepseek-ai/deepseek-coder-6.7b-base" 

def create_treuno_100b_model(vocab_size=32000):
    """Initializes the P&T Treuno 100B model architecture from scratch."""
    logging.info(f"Building P&T Treuno 100B architecture from scratch with vocab size {vocab_size}...")
    
    # Create configuration
    config_dict = TREUNO_100B_CONFIG.copy()
    config_dict["vocab_size"] = vocab_size
    config = AutoConfig.for_model("llama", **config_dict)
    
    # Initialize randomly initialized model (NOT pre-trained)
    # When using DeepSpeed Zero-3, this will lazily initialize parameters
    # to avoid blowing up CPU RAM.
    model = AutoModelForCausalLM.from_config(config)
    
    # Note: On a single machine, calculating parameters for a 100B model might instantly OOM the CPU
    # unless using proper lazy initialization (DeepSpeed Zero-3 / FSDP).
    try:
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Initialized P&T Treuno. Total Parameters: {total_params / 1e9:.2f} Billion")
    except Exception as e:
        logging.warning("Skipped exact parameter count (likely due to Zero-3 lazy param offloading).")
        
    return model

def get_datasets(tokenizer):
    """Loads and tokenizes the collected Elixir/General Code dataset."""
    logging.info("Loading processed dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run the scrapers first!")
        
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=8192, padding=False)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Split for train/validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42)
    return split_dataset['train'], split_dataset['test']

def main():
    logging.info("Initializing P&T Treuno 100B Pre-Training Pipeline")
    
    # 1. Load Tokenizer
    # For coding, you'd typically want a tokenizer trained on code like DeepSeek's or StarCoder's
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Setup Dataset
    train_dataset, eval_dataset = get_datasets(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 3. Create Model Architecture
    model = create_treuno_100b_model(vocab_size=len(tokenizer))
    
    # 4. Configure Training Arguments (Optimized for Multi-Node Cluster with DeepSpeed)
    training_args = TrainingArguments(
        output_dir="./treuno_100B_checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=4,     # Requires High VRAM
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,     
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=2000,
        logging_steps=100,
        learning_rate=1e-4,                # Lower LR for huge models
        weight_decay=0.1,
        warmup_steps=2000,
        lr_scheduler_type="cosine",
        bf16=True,                         # BF16 is mandatory for 100B stability (A100/H100)
        gradient_checkpointing=True,
        report_to="wandb",                 # Use Weights & Biases for monitoring
        deepspeed="deepspeed_zero3.json",  # Mandatory DeepSpeed Config
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # 6. Begin Pre-training
    logging.info("Starting distributed pre-training for P&T Treuno 100B...")
    trainer.train()
    
    # 7. Save final model
    trainer.save_model("./treuno_100B_final")
    logging.info("Pre-training complete! Model saved.")

if __name__ == "__main__":
    main()
