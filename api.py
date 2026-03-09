import os
import secrets
import uvicorn
import torch
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
MODEL_DIR = "./treuno_125M_final"
TOKENIZER_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
PORT = 8000

# 1. Generate an absolutely secure, random API key when the server starts
# This ensures ONLY authorized users can access the P&T Treuno brain.
API_KEY = secrets.token_hex(32)

app = FastAPI(title="P&T Treuno API", description="Secure Code Generation API")

# Setup Global variables for model
tokenizer = None
model = None

# --- Startup Event to Load Model ---
@app.on_event("startup")
def load_model():
    global tokenizer, model
    
    print("\n" + "="*50)
    print(f"YOUR AUTO-GENERATED API KEY:")
    print(f"    {API_KEY}")
    print("   (Save this! You must pass it in the `Authorization` header!)")
    print("="*50 + "\n")
    
    print(f"Loading P&T Treuno 125M from {MODEL_DIR}...")
    
    if not os.path.exists(MODEL_DIR):
        raise RuntimeError(f"Error: Model directory not found at {MODEL_DIR}. Train it first!")
        
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print("API is Ready for Requests!\n")

# --- Security Dependency ---
async def verify_api_key(authorization: str = Header(None)):
    """Verifies that the request has the correct auto-generated API key."""
    if authorization is None:
        raise HTTPException(status_code=401, detail="API Key missing. Pass 'Authorization: <key>'")
    
    # Strip "Bearer " if someone passes it like an OAuth token
    key = authorization.replace("Bearer ", "").strip()
    
    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API Key! Access Denied.")

# --- API Request Schema ---
class CodeGenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    # ZERO HALLUCINATION settings by default
    temperature: float = 0.01  
    top_p: float = 0.90

class CodeGenerationResponse(BaseModel):
    prompt: str
    generated_code: str

# --- API Endpoint ---
@app.post("/v1/completions", response_model=CodeGenerationResponse, dependencies=[Depends(verify_api_key)])
async def generate_code(request: CodeGenerationRequest):
    """
    Takes an input prompt and generates highly deterministic code 
    using the P&T Treuno AI model.
    """
    global tokenizer, model
    
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
        
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        # Generation with highly strict parameters (reducing hallucination)
        # Using temperature near 0 makes the model highly deterministic.
        do_sample = request.temperature > 0.05
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if do_sample else None,
            top_p=request.top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the output, stripping the original prompt from the text
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return CodeGenerationResponse(
            prompt=request.prompt,
            generated_code=generated_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting secure P&T Treuno FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
