import time
import schedule
import logging
from db.schema import db_schema

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

RETENTION_DAYS = 90

def purge_old_logs():
    logger.info(f"Running retention policy scan for logs older than {RETENTION_DAYS} days...")
    try:
        deleted = db_schema.delete_older_than(RETENTION_DAYS)
        logger.info(f"Retention policy applied successfully. Dropped {deleted} rows.")
    except Exception as e:
        logger.error(f"Error executing retention policy: {e}")

def run_worker():
    logger.info("Initializing retention policy worker...")
    # Schedule to run once a day
    schedule.every().day.at("02:00").do(purge_old_logs)
    
    # Run once at startup
    purge_old_logs()

    while True:
        schedule.run_pending()
        time.sleep(3600)

if __name__ == "__main__":
    run_worker()
