from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "aleehydar/clinicflow-llama-3.2-3b-medical"

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER,
            is_trainable=False,
        )
        model.eval()

def generate_soap(symptoms: str) -> str:
    """
    Generates a structured SOAP note from given symptoms.
    Uses the pre-loaded model with a 500 token maximum output.
    """
    global model, tokenizer
    
    if model is None:
        load_model()
    
    # Constructing a basic prompt depending on how the model was tuned.
    prompt = f"Patient Symptoms: {symptoms}\n\nGenerate a detailed SOAP (Subjective, Objective, Assessment, Plan) note for these symptoms:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Run the generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text if necessary
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
        
    return generated_text
