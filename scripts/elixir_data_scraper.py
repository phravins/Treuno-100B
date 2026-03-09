import os
import json
import time
import requests
import base64
from datetime import datetime

# GitHub API rate limits mean we need a token for serious scraping.
# For the prototype, we will run without one, but you should set GITHUB_TOKEN environment variable.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

HEADERS = {
    "Accept": "application/vnd.github.v3+json"
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

DATASET_FILE = os.path.join(DATASET_DIR, "elixir_raw_dataset.jsonl")

def search_elixir_repos(min_stars=50, limit=20):
    """Search for top Elixir repositories."""
    print(f"[*] Searching for Elixir repositories with >{min_stars} stars...")
    url = f"https://api.github.com/search/repositories?q=language:elixir+stars:>{min_stars}&sort=stars&order=desc&per_page={limit}"
    
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        print(f"[!] Error fetching repos: {response.status_code} - {response.text}")
        return []

def get_repo_files(repo_full_name, path=""):
    """Recursively get all .ex and .exs files in a repository."""
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    response = requests.get(url, headers=HEADERS)
    files = []
    
    if response.status_code == 200:
        contents = response.json()
        if not isinstance(contents, list):
            contents = [contents]
            
        for item in contents:
            if item["type"] == "file" and (item["name"].endswith(".ex") or item["name"].endswith(".exs")):
                files.append(item)
            elif item["type"] == "dir":
                # Sleep slightly to avoid aggressive rate limiting
                time.sleep(0.5)
                files.extend(get_repo_files(repo_full_name, item["path"]))
    return files

def download_file_content(download_url):
    """Download the raw content of a file."""
    if not download_url:
        return None
    response = requests.get(download_url)
    if response.status_code == 200:
        return response.text
    return None

def update_dataset():
    """Main function to run the auto-updater for the dataset."""
    print(f"[{datetime.now()}] Starting Elixir dataset update for P&T Treuno 100B pre-training...")
    
    repos = search_elixir_repos(min_stars=500, limit=5) # Start small for prototype
    
    total_files_collected = 0
    
    with open(DATASET_FILE, 'a', encoding='utf-8') as f:
        for repo in repos:
            repo_name = repo["full_name"]
            print(f"[*] Processing repository: {repo_name}")
            
            # Get all elixir files
            files = get_repo_files(repo_name)
            print(f"    Found {len(files)} Elixir files in {repo_name}.")
            
            for file_info in files:
                content = download_file_content(file_info.get("download_url"))
                if content:
                    # Save in JSONL format for easy loading by HuggingFace datasets
                    # Format: {"text": "source_code", "meta": {"repo": "xx", "path": "yy"}}
                    data_entry = {
                        "text": content,
                        "meta": {
                            "repo": repo_name,
                            "file_path": file_info.get("path"),
                            "url": file_info.get("html_url"),
                            "collected_at": datetime.now().isoformat()
                        }
                    }
                    f.write(json.dumps(data_entry) + "\n")
                    total_files_collected += 1
                
                # Small sleep to respect rate limits
                time.sleep(0.2)
                
            print(f"[*] Finished processing {repo_name}.")
    
    print(f"[{datetime.now()}] Dataset update complete! Added {total_files_collected} new Elixir files to dataset.")

if __name__ == "__main__":
    update_dataset()
