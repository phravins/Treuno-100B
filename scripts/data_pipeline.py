import os
import json
import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW_ELIXIR_FILES = [
    os.path.join(DATASET_DIR, "elixir_raw_dataset.jsonl"),
    os.path.join(DATASET_DIR, "elixir_docs_dataset.jsonl")
]
GENERAL_FILE = os.path.join(DATASET_DIR, "general_code_dataset.jsonl")
PROCESSED_FILE = os.path.join(DATASET_DIR, "processed_dataset.jsonl")

# Multiplier for Elixir data to ensure the model heavily prioritizes Elixir
ELIXIR_OVERSAMPLE_MULTIPLIER = 10

# We will use the DeepSeek-Coder tokenizer as a base since it is highly optimized 
# for code, but since P&T Treuno is built from scratch we train our own BPE tokenizer
# in a real 100B parameter scenario. For this pipeline, we will demonstrate 
# loading a standard code-optimized tokenizer.
TOKENIZER_NAME = "deepseek-ai/deepseek-coder-6.7b-base" 

def clean_elixir_code(text):
    """
    Basic data cleaning rules for Elixir code.
    Removes massive chunks of commented out code or empty lines, 
    but preserves docstrings.
    """
    if not text:
        return ""
    
    # Remove excessive blank lines
    lines = text.split('\n')
    cleaned_lines = []
    blank_streak = 0
    
    for line in lines:
        if not line.strip():
            blank_streak += 1
            if blank_streak > 2:
                continue # Skip more than 2 consecutive blank lines
        else:
            blank_streak = 0
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def process_datasets():
    logging.info("Starting data cleaning pipeline for P&T Treuno 100B...")
    
    try:
        # Note: In a real environment you'd run tokenizer.train_new_from_iterator()
        # if building a fully custom tokenizer for P&T Treuno. Here we load a robust
        # foundation tokenizer for the training loop proof-of-concept.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    except Exception as e:
        logging.warning(f"Could not load {TOKENIZER_NAME}. Error: {e}")
        logging.warning("Please run `pip install transformers` and authenticate if necessary.")
        tokenizer = None
    
    total_processed = 0
    total_raw = 0
    total_elixir_written = 0
    total_general_written = 0
    
    with open(PROCESSED_FILE, 'w', encoding='utf-8') as outfile:
        # 1. Process General Code Dataset
        if os.path.exists(GENERAL_FILE):
            logging.info(f"Processing Multi-Language Context: {GENERAL_FILE}")
            with open(GENERAL_FILE, 'r', encoding='utf-8') as infile:
                for line in infile:
                    total_raw += 1
                    try:
                        entry = json.loads(line)
                        cleaned_text = clean_elixir_code(entry.get("text", "")) # Standard clean
                        if len(cleaned_text.strip()) < 10: continue
                        entry["text"] = cleaned_text
                        outfile.write(json.dumps(entry) + "\n")
                        total_processed += 1
                        total_general_written += 1
                    except Exception:
                        pass
        
        # 2. Process and Oversample Elixir Dataset
        for file_path in RAW_ELIXIR_FILES:
            if not os.path.exists(file_path):
                logging.warning(f"Raw Elixir dataset file not found: {file_path}")
                continue
                
            logging.info(f"Processing & Oversampling Elixir Data: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    total_raw += 1
                    try:
                        entry = json.loads(line)
                        cleaned_text = clean_elixir_code(entry.get("text", ""))
                        if len(cleaned_text.strip()) < 10:
                            continue
                            
                        # Tokenization (Optional Check step)
                        if tokenizer:
                            tokens = tokenizer.encode(cleaned_text)
                            entry["token_count"] = len(tokens)
                            
                        entry["text"] = cleaned_text
                        
                        # Oversample the Elixir data to force model prioritization
                        for _ in range(ELIXIR_OVERSAMPLE_MULTIPLIER):
                            outfile.write(json.dumps(entry) + "\n")
                            total_processed += 1
                            total_elixir_written += 1
                            
                    except json.JSONDecodeError:
                        logging.error("Failed to parse JSON line. Skipping.")
                    except Exception as e:
                        logging.error(f"Error processing entry: {e}")
                        
    logging.info(f"Pipeline complete! Raw entries processed: {total_raw} | Total Output Sequences: {total_processed}")
    logging.info(f"General Programming Sequences: {total_general_written} | Elixir Sequences (Oversampled): {total_elixir_written}")
    logging.info(f"Saved to: {PROCESSED_FILE}")

if __name__ == "__main__":
    process_datasets()
