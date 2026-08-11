import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_URL = "https://router.huggingface.co/hf-inference/models/aleehydar/clinicflow-llama-3.2-3b-medical"

def load_model():
    """
    Placeholder since we are using the remote Inference API.
    """
    pass

def generate_soap(symptoms: str) -> str:
    prompt = (
        "### Instruction:\n"
        "Generate a SOAP clinical note using ONLY the information explicitly provided below. "
        "Do NOT invent vital signs, medications, past medical history, family history, social history, "
        "or physical exam findings that are not stated in the input. "
        "If a section has no information provided, write 'Not documented' for that section instead of fabricating content.\n\n"
        f"### Input:\n{symptoms}\n\n### Response:\n"
    )
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
