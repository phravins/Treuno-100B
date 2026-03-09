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

# --- CONFIGURATION FOR P&T TREUNO 125M (FREE TIER) ---
# We define an architecture based on Llama scaled down to ~125M parameters
# so it can be trained for free on Google Colab or Kaggle.
TREUNO_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 768,              # Standard dimension for small models
    "intermediate_size": 3072,       # SwiGLU intermediate size
    "num_hidden_layers": 12,         # 12 Layers (like GPT2-Small)
    "num_attention_heads": 12,       # 12 Attention heads
    "num_key_value_heads": 12,       # Standard Multi-Head Attention
    "max_position_embeddings": 2048, # 2048 Context window limit
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
    "vocab_size": 32000              # Standard tokenizer vocab size
}

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "processed_dataset.jsonl")
# Using DeepSeek base tokenizer as placeholder, must match vocab_size in config
TOKENIZER_NAME = "deepseek-ai/deepseek-coder-6.7b-base" 

def create_treuno_125m_model(vocab_size=32000):
    """Initializes the P&T Treuno 125M model architecture completely from scratch."""
    logging.info(f"Building P&T Treuno 125M architecture from scratch with vocab size {vocab_size}...")
    
    # Create configuration
    config_dict = TREUNO_CONFIG.copy()
    config_dict["vocab_size"] = vocab_size
    config = AutoConfig.for_model("llama", **config_dict)
    
    # Initialize randomly initialized model (NOT pre-trained)
    # When using DeepSpeed Zero-3, this will lazily initialize parameters
    model = AutoModelForCausalLM.from_config(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Initialized P&T Treuno. Total Parameters: {total_params / 1e9:.2f} Billion")
    return model

def get_datasets(tokenizer):
    """Loads and tokenizes the collected Elixir dataset."""
    logging.info("Loading processed Elixir dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run the scrapers first!")
        
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=8192, padding=False)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Split for train/validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    return split_dataset['train'], split_dataset['test']

def main():
    logging.info("Initializing P&T Treuno 100B Pre-Training Pipeline")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Setup Dataset
    train_dataset, eval_dataset = get_datasets(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 3. Create Model Architecture
    model = create_treuno_125m_model(vocab_size=len(tokenizer))
    
    # 4. Configure Training Arguments (Optimized for single 16GB GPU like T4)
    training_args = TrainingArguments(
        output_dir="./treuno_125M_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=100,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        fp16=True,                         # T4 GPUs support fp16 (not bf16)
        gradient_checkpointing=True,
        report_to="none"                   # Change to wandb in production
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
    logging.info("Starting pre-training for P&T Treuno 125M...")
    trainer.train()
    
    # 7. Save final model
    trainer.save_model("./treuno_125M_final")
    logging.info("Pre-training complete! Model saved.")

if __name__ == "__main__":
    main()
