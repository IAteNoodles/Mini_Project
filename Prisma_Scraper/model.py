from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_name = "unsloth/mistral-7b-v0.2-bnb-4bit"  # 4-bit quantized Mistral 7B model
    device = "cuda" 
    
    print(f"Loading tokenizer and model on {device} ...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with 4-bit quantization and CUDA acceleration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          # Automatically place model parts on GPU/CPU
        load_in_4bit=True,          # Load model with 4-bit quantization
        dtype=torch.float16,  # Mixed precision for better performance
    )
    
    model.to(device)
    
    # Input prompt for generation
    prompt = "Explain me Arch Linux Core in as long as possible"
    
    # Tokenize prompt and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate output tokens with a limit on max length
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,  # deterministic (greedy) decoding
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode and print generated text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated Text:\n{result}")

if __name__ == "__main__":
    main()
