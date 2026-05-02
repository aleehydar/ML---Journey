# Corpus Update Protocol

## Scope
How to add new PDF laws or update existing ones without incurring downtime.

## Procedure
1. **Staging the Data:** Place new `.pdf` files into a temporary server directory `data_new/`.
2. **Execute Ingestion:** Run `python ingest.py --data_dir data_new/ --output vectorstore_new/` (Assuming args are supported, otherwise rename folders temporarily).
3. **Atomic Swap:**
   - Stop web traffic temporarily or queue requests if necessary (Optional).
   - Swap directories on disk: 
     ```bash
     mv vectorstore vectorstore_old
     mv vectorstore_new vectorstore
     ```
4. **Hot Reload:** Restart FastAPI `docker-compose restart legal-ai` to load the new FAISS index into memory.
5. **Validation:** Query `/documents` to ensure the new files are correctly indexed and mapped.
