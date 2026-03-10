import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_DIR = "./treuno_125M_final"
TOKENIZER_NAME = "deepseek-ai/deepseek-coder-6.7b-base"

def main():
    print(f"Loading P&T Treuno 125M from {MODEL_DIR}...")
    
    # Check if the model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at {MODEL_DIR}.")
        print("Please make sure you have successfully run train.py first!")
        return
        
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load the custom trained model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    print("\n" + "="*50)
    print("P&T Treuno 125M is Online")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    # Interactive Chat Loop
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ['quit', 'exit']:
                break
                
            if not prompt.strip():
                continue
                
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            print("\nTreuno: ", end="", flush=True)
            
            # Streaming the output is complex, so we simply generate it and print
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,     # How many tokens to write
                temperature=0.7,        # Creativity (higher = more random)
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated text, removing the prompt from the output
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(generated_text)
            
        except KeyboardInterrupt:
            break
            
    print("\nShutting down P&T Treuno.")

if __name__ == "__main__":
    main()
