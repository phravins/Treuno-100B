import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# Configuration
MODEL_DIR = "./treuno_125M_final"
TOKENIZER_NAME = "deepseek-ai/deepseek-coder-6.7b-base"

# Initialize FastAPI app
app = FastAPI(title="P&T Treuno 125M Inference API", description="Automatic API to query the Treuno model.")

# Global variables for model and tokenizer
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9

@app.on_event("startup")
def load_startup():
    global model, tokenizer
    print(f"Loading P&T Treuno 125M from {MODEL_DIR}...")
    
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at {MODEL_DIR}.")
        return
        
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load your custom trained model!
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print("\n" + "="*50)
    print("🧠 P&T Treuno 125M API is Online! 🧠")
    print("="*50 + "\n")

@app.post("/api/generate")
def generate_text(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Make sure you trained the model first.")
        
    # Tokenize input
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated text (excluding the prompt from output)
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return {
        "prompt": req.prompt, 
        "treuno_response": generated_text
    }

if __name__ == "__main__":
    # Run the API server automatically on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
