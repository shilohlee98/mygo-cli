#!/usr/bin/env python3
"""MyGo image search tool - supports fuzzy matching and semantic search"""

import json
import logging
import os
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from difflib import SequenceMatcher

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["SAFETENSORS_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
from sentence_transformers import SentenceTransformer

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

API_BASE = "https://mygo.miyago9267.com/api/v1/images"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, ".mygo_cache.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, ".mygo_embeddings.npy")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
PAGE_LIMIT = 100


def fetch_all_images():
    """Fetch all image data from API"""
    all_images = []
    page = 1
    while True:
        url = f"{API_BASE}?page={page}&limit={PAGE_LIMIT}&order=id"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
        all_images.extend(data["data"])
        if not data["meta"]["hasNext"]:
            break
        page += 1
        print(f"\rLoading... {len(all_images)}/{data['meta']['total']}", end="", flush=True)
    print(f"\rLoaded {len(all_images)} images")
    return all_images


def load_images():
    """Load image data (use cache if available)"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            images = json.load(f)
        print(f"Loaded {len(images)} images from cache")
        return images
    images = fetch_all_images()
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False)
    return images


def load_model():
    """Load sentence-transformers model"""
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    return model


def build_embeddings(model, images):
    """Build or load embeddings for all alt texts"""
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        if len(embeddings) == len(images):
            print("Loaded embeddings from cache")
            return embeddings
        print("Image count changed, rebuilding embeddings...")
    alts = [img["alt"] for img in images]
    print(f"Building {len(alts)} embeddings (slower on first run)...")
    embeddings = model.encode(alts, show_progress_bar=True, normalize_embeddings=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    print("Embeddings cached")
    return embeddings


def semantic_search(query, model, embeddings, images, top_n=5):
    """Semantic search"""
    query_emb = model.encode([query], normalize_embeddings=True)
    scores = (query_emb @ embeddings.T)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(float(scores[i]), images[i]) for i in top_indices]


def fuzzy_search(query, images, top_n=5):
    """Fuzzy string search"""
    results = []
    query_lower = query.lower()
    for img in images:
        alt = img["alt"]
        alt_lower = alt.lower()
        if query_lower == alt_lower:
            score = 2.0
        elif query_lower in alt_lower or alt_lower in query_lower:
            score = 1.5 + SequenceMatcher(None, query_lower, alt_lower).ratio()
        else:
            score = SequenceMatcher(None, query_lower, alt_lower).ratio()
        results.append((score, img))
    results.sort(key=lambda x: (-x[0], -x[1]["popularity"]))
    return results[:top_n]


def copy_image_to_clipboard(url):
    """Download image and copy to macOS clipboard"""
    from urllib.parse import quote, urlparse, urlunparse
    parsed = urlparse(url)
    safe_url = urlunparse(parsed._replace(path=quote(parsed.path, safe="/")))
    req = urllib.request.Request(safe_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CTX) as resp:
        img_data = resp.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(img_data)
        tmp_path = f.name
    try:
        # Use hex-safe temp path to avoid encoding issues
        safe_path = tempfile.mktemp(suffix=".jpg", dir=tempfile.gettempdir())
        os.rename(tmp_path, safe_path)
        tmp_path = safe_path
        script = f'set the clipboard to (read (POSIX file "{tmp_path}") as JPEG picture)'
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        return True
    finally:
        os.unlink(tmp_path)


def print_results(query, results, mode):
    print(f'\nResults for "{query}" ({mode}):\n')
    for i, (score, img) in enumerate(results, 1):
        print(f"  {i}. {img['alt']}")
        print(f"     Score: {score:.4f} | Episode: {img['episode']} | Popularity: {img['popularity']}")
        print(f"     {img['url']}\n")


def main():
    images = load_images()
    model = load_model()
    embeddings = build_embeddings(model, images)
    print()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = semantic_search(query, model, embeddings, images)
        print_results(query, results, "semantic")
        return

    # Interactive mode
    print("Search MyGo images (semantic by default, prefix /f for fuzzy search, q to quit)")
    print("After results, enter a number (1-5) to copy that image to clipboard\n")
    last_results = []
    while True:
        try:
            raw = input("Search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() == "q":
            break
        # Copy image by number
        if raw.isdigit() and last_results:
            idx = int(raw)
            if 1 <= idx <= len(last_results):
                _, img = last_results[idx - 1]
                print(f"  Copying \"{img['alt']}\"...")
                try:
                    copy_image_to_clipboard(img["url"])
                    print("  Copied to clipboard!\n")
                except Exception as e:
                    print(f"  Failed to copy: {e}\n")
            else:
                print(f"  Invalid number, enter 1-{len(last_results)}\n")
            continue
        if raw.startswith("/f "):
            query = raw[3:].strip()
            last_results = fuzzy_search(query, images)
            print_results(query, last_results, "fuzzy")
        else:
            last_results = semantic_search(raw, model, embeddings, images)
            print_results(raw, last_results, "semantic")


if __name__ == "__main__":
    main()
