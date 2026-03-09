import schedule
import time
import subprocess
import os
import sys
from datetime import datetime

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GITHUB_SCRAPER = os.path.join(SCRIPT_DIR, "elixir_data_scraper.py")
HEXDOCS_SCRAPER = os.path.join(SCRIPT_DIR, "hexdocs_scraper.py")

def run_github_scraper():
    print(f"[{datetime.now()}] Triggering scheduled GitHub scrape...")
    subprocess.run([sys.executable, GITHUB_SCRAPER], check=False)

def run_hexdocs_scraper():
    print(f"[{datetime.now()}] Triggering scheduled HexDocs scrape...")
    subprocess.run([sys.executable, HEXDOCS_SCRAPER], check=False)

def run_all_updates():
    print(f"\n[{datetime.now()}] === STARTING ELIXIR DATASET AUTO-UPDATE ===")
    run_hexdocs_scraper()
    run_github_scraper()
    print(f"[{datetime.now()}] === FINISHED ELIXIR DATASET AUTO-UPDATE ===\n")

if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting P&T Treuno 100B Auto-Update Scheduler.")
    print("Updates are scheduled to run daily at 00:00.")
    print("Press Ctrl+C to exit.")
    
    # Run once on startup to ensure initial dataset exists
    print("Running initial population...")
    run_all_updates()
    
    # Schedule daily updates
    schedule.every().day.at("00:00").do(run_all_updates)
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60) # check every minute
    except KeyboardInterrupt:
        print("\nScheduler terminated by user.")
