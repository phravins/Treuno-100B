#!/usr/bin/env python3
import os
import json
import logging
from datasets import load_dataset
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
GENERAL_CODE_FILE = os.path.join(DATASET_DIR, "general_code_dataset.jsonl")

# We use the open 'starcoderdata' via streaming to access massive multi-language 
# code data without requiring terabytes of local storage upfront.
DATASET_NAME = "bigcode/starcoderdata"

def download_general_code(target_samples=100000):
    """
    Downloads a subset of general programming languages to train the base logic.
    In a real 100B training run, this would stream continuously during training.
    """
    logging.info(f"Starting download of multi-language coding dataset from {DATASET_NAME}...")
    logging.info(f"Target sample size for prototype: {target_samples} sequences.")
    
    try:
        # We stream the dataset to avoid massive memory/disk overhead 
        # for downloading billions of tokens at once.
        dataset = load_dataset(DATASET_NAME, streaming=True, split="train")
        
        saved_count = 0
        with open(GENERAL_CODE_FILE, 'w', encoding='utf-8') as f:
            for example in dataset:
                # Basic filtering
                if 'content' not in example or not example['content']:
                    continue
                    
                entry = {
                    "text": example['content'],
                    "meta": {
                        "repository": example.get('repository_name', 'unknown'),
                        "language": example.get('language', 'multi'),
                        "source": "starcoderdata"
                    }
                }
                
                f.write(json.dumps(entry) + "\n")
                saved_count += 1
                
                if saved_count % 5000 == 0:
                    logging.info(f"Downloaded {saved_count}/{target_samples} multi-language sequences...")
                    
                if saved_count >= target_samples:
                    break
        
        logging.info(f"Successfully downloaded {saved_count} general coding sequences.")
        logging.info(f"Saved to: {GENERAL_CODE_FILE}")
        
    except Exception as e:
        logging.error(f"Failed to download multi-language dataset: {e}")
        logging.error("Ensure you have `transformers` and `datasets` installed, and sufficient disk space.")

if __name__ == "__main__":
    download_general_code()
