# ClinicFlow - AI Clinical Assistant

ClinicFlow is an AI-powered clinical assistant for Pakistani healthcare workers, generating structured SOAP notes from patient symptoms and performing basic rule-based triage. It wires together a fine-tuned HuggingFace LLaMA model and runs as a lightweight FastAPI service.

## Architecture

```text
       +----------------------------------------------------+
       |                  EC2 Instance                      |
       |                                                    |
       |  +------------------+         +-----------------+  |
       |  | Attrition API    |         | ClinicFlow API  |  |
------>|  | Port 8000        |         | Port 8001       |  |
       |  +------------------+         +-----------------+  |
       |                                 |         ^        |
       |                                 v         |        |
       |                        +-----------------------+   |
       |                        | aleehydar/clinicflow  |   |
       |                        | -llama-3.2-3b-medical |   |
       |                        +-----------------------+   |
       +----------------------------------------------------+
```

## How to Run Locally

1. **Clone the repository and enter the directory:**
   ```bash
   cd clinicflow
   ```

2. **Run with Docker Compose (Recommended):**
   ```bash
   docker-compose up --build
   ```

3. **Or run manually with Python:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8001
   ```

4. **Access the Application:**
   Open your browser and navigate to: `http://localhost:8001`

## Live Deployment

Once deployed, the live API and UI will be available at:
`http://3.6.133.108:8001`

*(Note: The attrition API continues to run on port 8000).*

## Disclaimer

**Not a substitute for qualified medical professionals.** 
This tool is for educational use only. AI-generated notes and triage priorities must be reviewed and verified by certified healthcare practitioners.
