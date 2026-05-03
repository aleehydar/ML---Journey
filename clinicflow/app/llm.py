import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/aleehydar/clinicflow-llama-3.2-3b-medical"

def load_model():
    """
    Placeholder since we are using the remote Inference API.
    """
    pass

def generate_soap(symptoms: str) -> str:
    prompt = f"### Instruction:\nGenerate a SOAP clinical note\n\n### Input:\n{symptoms}\n\n### Response:\n"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(
        API_URL, 
        headers=headers, 
        json={"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    )
    
    result = response.json()
    if isinstance(result, list):
        return result[0].get("generated_text", "").split("### Response:")[-1].strip()
        
    return "Model loading, please try again in 30 seconds."
