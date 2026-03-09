import os
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

DATASET_FILE = os.path.join(DATASET_DIR, "elixir_docs_dataset.jsonl")

# Base URL for Hex API
HEX_API_URL = "https://hex.pm/api/packages"

def get_top_packages(limit=50):
    """Fetch top Elixir packages from Hex.pm."""
    print(f"[*] Fetching top {limit} packages from Hex.pm...")
    # The API sorts by downloads by default, so we just get the first page
    response = requests.get(f"{HEX_API_URL}?sort=downloads")
    
    if response.status_code == 200:
        packages = response.json()
        return [pkg['name'] for pkg in packages[:limit]]
    else:
        print(f"[!] Failed to fetch packages: {response.status_code}")
        return []

def scrape_hex_docs(package_name):
    """Scrape the main documentation page for a package."""
    url = f"https://hexdocs.pm/{package_name}/readme.html"
    print(f"    Scraping docs for {package_name}...")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # The main content is usually in a div with id="content" or similar
            # For simplicity, we extract all text and clean it up slightly
            content_div = soup.find('div', id='content')
            
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
            else:
                # Fallback to body tag
                text = soup.body.get_text(separator='\n', strip=True)
                
            return text
        else:
            return None
    except Exception as e:
        print(f"      [!] Error scraping {package_name}: {e}")
        return None

def update_docs_dataset():
    """Main execution function for the Hex scraper."""
    print(f"[{datetime.now()}] Starting HexDocs dataset update for P&T Treuno 100B...")
    
    packages = get_top_packages(limit=20) # Prototype limit
    total_docs_collected = 0
    
    with open(DATASET_FILE, 'a', encoding='utf-8') as f:
        for pkg in packages:
            doc_text = scrape_hex_docs(pkg)
            
            if doc_text:
                data_entry = {
                    "text": doc_text,
                    "meta": {
                        "source": "hexdocs",
                        "package": pkg,
                        "collected_at": datetime.now().isoformat()
                    }
                }
                f.write(json.dumps(data_entry) + "\n")
                total_docs_collected += 1
                
            time.sleep(1) # Be nice to the hexdocs server
            
    print(f"[{datetime.now()}] HexDocs update complete! Added {total_docs_collected} documentation files to dataset.")

if __name__ == "__main__":
    update_docs_dataset()
