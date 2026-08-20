#!/usr/bin/env python3
"""MyGo image search tool - supports fuzzy matching and semantic search"""

import json
import logging
import os
import re
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

BASE_DIR = Path(__file__).resolve().parent


def resolve_writable_dir(requested, fallback):
    """Use the requested cache directory when writable, otherwise use a safe fallback."""
    requested = Path(requested).expanduser()
    fallback = Path(fallback).expanduser()
    for candidate in (requested, fallback):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".mygo-write-test-", dir=candidate):
                pass
            return candidate
        except OSError:
            continue
    raise OSError(f"Neither cache directory is writable: {requested}, {fallback}")


REQUESTED_DATA_DIR = Path(
    os.environ.get(
        "MYGO_DATA_DIR",
        Path.home() / "Library" / "Application Support" / "MyGo",
    )
).expanduser()
DATA_DIR = resolve_writable_dir(
    REQUESTED_DATA_DIR,
    Path(tempfile.gettempdir()) / f"MyGo-{os.getuid()}",
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

API_BASE = "https://mygo.miyago9267.com/api/v1/images"
CACHE_FILE = DATA_DIR / "images.json"
MODEL_NAME = "multilingual-e5-small"
MODEL_SLUG = MODEL_NAME.replace("/", "__")
MODEL_REPOSITORY_URL = (
    "https://huggingface.co/Xenova/"
    f"{MODEL_NAME}/resolve/main"
)
MODEL_DIMENSION = 384
MODEL_QUERY_PREFIX = "query: "
MODEL_DOCUMENT_PREFIX = "passage: "
MODEL_POOLING = "mean"
MODEL_PAD_ID = 1
MODEL_PAD_TOKEN = "<pad>"
EMBEDDINGS_FILE = DATA_DIR / f"embeddings.{MODEL_SLUG}.npy"
ONNX_MODEL_DIR = DATA_DIR / "models" / MODEL_SLUG
MODEL_BACKEND = os.environ.get("MYGO_MODEL_BACKEND", "onnx").strip().lower()
ONNX_QUANTIZATION = os.environ.get("MYGO_ONNX_QUANTIZATION", "auto").strip().lower()
ONNX_PROVIDER = os.environ.get("MYGO_ONNX_PROVIDER", "CPUExecutionProvider").strip()
VALID_ONNX_QUANTIZATION = {
    "auto",
    "int8",
    "arm64",
    "avx2",
    "avx512",
    "avx512_vnni",
}
PAGE_LIMIT = 100
SHORT_ENGLISH_QUERY_ALIASES = {
    "hi": "妳好",
    "hello": "妳好",
}
StatusCallback = Callable[[str], None]


def report(status: StatusCallback | None, message: str):
    """Send progress to a CLI or GUI without assuming stdout is available."""
    if status is not None:
        status(message)


def ensure_data_dir():
    """Create the macOS-writable cache location on demand."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_all_images(status: StatusCallback | None = print):
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
        report(status, f"Loading images... {len(all_images)}/{data['meta']['total']}")
    report(status, f"Loaded {len(all_images)} images")
    return all_images


def load_images(status: StatusCallback | None = print):
    """Load image data (use cache if available)"""
    ensure_data_dir()
    if CACHE_FILE.exists():
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            images = json.load(f)
        report(status, f"Loaded {len(images)} images from cache")
        return images
    images = fetch_all_images(status)
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False)
    return images


def load_model(status: StatusCallback | None = print):
    """Load the lightweight ONNX sentence encoder."""
    ensure_data_dir()
    if DATA_DIR != REQUESTED_DATA_DIR:
        report(status, f"Data directory is not writable; using {DATA_DIR}")
    report(status, f"Loading model {MODEL_NAME} [{MODEL_BACKEND}]...")
    if MODEL_BACKEND != "onnx":
        raise ValueError("MYGO_MODEL_BACKEND must be 'onnx'")
    try:
        return load_onnx_model(status)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the ONNX model. Make sure the data directory is writable "
            f"({DATA_DIR}) and the first launch has network access. "
            f"Original error: {exc}"
        ) from exc


def resolve_onnx_model_file_name():
    """Resolve the prebuilt ONNX file while accepting legacy settings."""
    if ONNX_QUANTIZATION in {"", "0", "false", "no", "none", "off"}:
        return "model.onnx"
    if ONNX_QUANTIZATION not in VALID_ONNX_QUANTIZATION:
        raise ValueError("MYGO_ONNX_QUANTIZATION must be 'auto', 'int8', or 'off'")
    return "model_int8.onnx"


class OnnxSentenceEncoder:
    """Small SentenceTransformer-compatible encoder without importing PyTorch."""

    def __init__(
        self,
        model_path,
        tokenizer_path,
        provider=ONNX_PROVIDER,
        embedding_dimension=MODEL_DIMENSION,
        pooling=MODEL_POOLING,
        query_prefix=MODEL_QUERY_PREFIX,
        document_prefix=MODEL_DOCUMENT_PREFIX,
        pad_id=MODEL_PAD_ID,
        pad_token=MODEL_PAD_TOKEN,
        max_length=128,
    ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[provider],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if self.tokenizer.padding is None:
            self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        if self.tokenizer.truncation is None:
            self.tokenizer.enable_truncation(max_length=max_length)
        self.embedding_dimension = embedding_dimension
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def get_backend(self):
        return "onnx"

    def encode(
        self,
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=False,
        is_query=False,
        is_document=False,
    ):
        del show_progress_bar
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if is_query and is_document:
            raise ValueError("A batch cannot be both query and document text")
        if is_query and self.query_prefix:
            texts = [f"{self.query_prefix}{text}" for text in texts]
        elif is_document and self.document_prefix:
            texts = [f"{self.document_prefix}{text}" for text in texts]
        batches = []
        for start in range(0, len(texts), batch_size):
            batches.append(
                self._encode_batch(
                    texts[start : start + batch_size],
                    normalize_embeddings,
                )
            )
        if not batches:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def _encode_batch(self, texts, normalize_embeddings):
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.asarray(
            [encoding.attention_mask for encoding in encodings],
            dtype=np.int64,
        )
        token_type_ids = np.asarray(
            [encoding.type_ids for encoding in encodings],
            dtype=np.int64,
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        inputs = {name: value for name, value in inputs.items() if name in self.input_names}
        token_embeddings = self.session.run(None, inputs)[0]
        if self.pooling == "cls":
            embeddings = token_embeddings[:, 0]
        elif self.pooling == "mean":
            expanded_mask = attention_mask[..., None].astype(token_embeddings.dtype)
            embeddings = (token_embeddings * expanded_mask).sum(axis=1)
            embeddings /= np.clip(expanded_mask.sum(axis=1), 1e-9, None)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings /= np.clip(norms, 1e-12, None)
        return embeddings


def download_model_asset(relative_path, destination, status: StatusCallback | None = print):
    """Atomically download one model asset into MyGo's private cache."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{MODEL_REPOSITORY_URL}/{relative_path}",
        headers={"User-Agent": "MyGo/0.4"},
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urllib.request.urlopen(request, context=SSL_CTX) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                next_update = 8 * 1024 * 1024
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_update and total:
                        report(status, f"Downloading model... {downloaded / total:.0%}")
                        next_update += 8 * 1024 * 1024
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_onnx_model(status: StatusCallback | None = print):
    """Load the cached ONNX model or download the prebuilt CPU assets."""
    model_dir = Path(ONNX_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file_name = resolve_onnx_model_file_name()
    model_path = model_dir / "onnx" / model_file_name
    tokenizer_path = model_dir / "tokenizer.json"
    regular_model = model_dir / "onnx" / "model.onnx"

    if not model_path.exists():
        try:
            quantized = model_file_name == "model_int8.onnx"
            description = "optimized int8 E5 ONNX model" if quantized else "E5 ONNX model"
            report(status, f"Downloading {description} (first run)...")
            download_model_asset(f"onnx/{model_file_name}", model_path, status)
        except Exception:
            if model_file_name == "model.onnx" or not regular_model.exists():
                raise
            model_path = regular_model
            report(status, "Optimized ONNX unavailable; using cached regular ONNX model...")

    if not tokenizer_path.exists():
        report(status, "Downloading tokenizer (first run)...")
        download_model_asset("tokenizer.json", tokenizer_path, status)

    report(status, f"Loading cached ONNX model ({model_path.name})...")
    return OnnxSentenceEncoder(
        model_path,
        tokenizer_path,
        embedding_dimension=MODEL_DIMENSION,
        pooling=MODEL_POOLING,
        query_prefix=MODEL_QUERY_PREFIX,
        document_prefix=MODEL_DOCUMENT_PREFIX,
        pad_id=MODEL_PAD_ID,
        pad_token=MODEL_PAD_TOKEN,
    )


def build_embeddings(model, images, status: StatusCallback | None = print, show_progress_bar=True):
    """Build or load embeddings for all alt texts"""
    ensure_data_dir()
    if EMBEDDINGS_FILE.exists():
        embeddings = np.load(EMBEDDINGS_FILE)
        if len(embeddings) == len(images):
            report(status, "Loaded embeddings from cache")
            return embeddings
        report(status, "Image count changed, rebuilding embeddings...")
    alts = [img["alt"] for img in images]
    report(status, f"Building {len(alts)} embeddings (slower on first run)...")
    embeddings = model.encode(
        alts,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        is_document=True,
    )
    np.save(EMBEDDINGS_FILE, embeddings)
    report(status, "Embeddings cached")
    return embeddings


def semantic_search(query, model, embeddings, images, top_n=5):
    """Semantic search"""
    semantic_query = normalize_semantic_query(query)
    query_emb = model.encode(
        [semantic_query],
        normalize_embeddings=True,
        is_query=True,
    )
    scores = (query_emb @ embeddings.T)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(float(scores[i]), images[i]) for i in top_indices]


def normalize_semantic_query(query):
    """Map ambiguous short English greetings to a stable Chinese search concept."""
    normalized = " ".join(query.casefold().split())
    return SHORT_ENGLISH_QUERY_ALIASES.get(normalized, query)


def literal_contains(container, needle):
    """Use ASCII word boundaries for English needles and substring matching otherwise."""
    if not needle:
        return False
    if needle.isascii() and any(character.isalpha() for character in needle):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])"
        return re.search(pattern, container, flags=re.IGNORECASE) is not None
    return needle in container


def lexical_similarity(query, text):
    """Return a normalized text match score for exact, substring, and typo matches."""
    query_lower = query.casefold()
    text_lower = text.casefold()
    if not query_lower or not text_lower:
        return 0.0
    ratio = SequenceMatcher(None, query_lower, text_lower).ratio()
    if query_lower == text_lower:
        return 1.0
    if literal_contains(text_lower, query_lower) or literal_contains(query_lower, text_lower):
        return 0.85 + 0.15 * ratio
    if query_lower.isascii() and query_lower.isalpha():
        english_words = re.findall(r"[a-z]+", text_lower)
        word_ratio = max(
            (SequenceMatcher(None, query_lower, word).ratio() for word in english_words),
            default=0.0,
        )
        return word_ratio if word_ratio >= 0.8 else 0.0
    return ratio


def fuzzy_search(query, images, top_n=5):
    """Fuzzy string search"""
    results = []
    for img in images:
        alt = img["alt"]
        score = lexical_similarity(query, alt)
        results.append((score, img))
    results.sort(key=lambda x: (-x[0], -x[1].get("popularity", 0)))
    return results[:top_n]


def hybrid_search(query, model, embeddings, images, top_n=5):
    """Blend multilingual semantic relevance with exact and fuzzy text matching."""
    semantic_query = normalize_semantic_query(query)
    query_emb = model.encode(
        [semantic_query],
        normalize_embeddings=True,
        is_query=True,
    )
    semantic_scores = (query_emb @ embeddings.T)[0]
    score_range = float(np.ptp(semantic_scores))
    if score_range > 1e-9:
        semantic_scores = (semantic_scores - semantic_scores.min()) / score_range
    else:
        semantic_scores = np.zeros_like(semantic_scores)

    lexical_scores = np.asarray(
        [lexical_similarity(query, image["alt"]) for image in images],
        dtype=np.float32,
    )
    query_lower = query.casefold()
    lexical_weight = 0.55 if len(query_lower) <= 4 else 0.35
    combined_scores = (
        (1.0 - lexical_weight) * semantic_scores
        + lexical_weight * lexical_scores
    )

    for index, image in enumerate(images):
        alt_lower = image["alt"].casefold()
        if query_lower == alt_lower:
            combined_scores[index] += 1.0
        elif literal_contains(alt_lower, query_lower):
            combined_scores[index] += 0.45
        elif literal_contains(query_lower, alt_lower):
            combined_scores[index] += 0.2

    top_indices = sorted(
        range(len(images)),
        key=lambda index: (
            float(combined_scores[index]),
            images[index].get("popularity", 0),
        ),
        reverse=True,
    )[:top_n]
    return [(float(combined_scores[index]), images[index]) for index in top_indices]


def route_search(query, model, embeddings, images, top_n=5):
    """Select hybrid, semantic, or fuzzy mode from the query prefix."""
    query = query.strip()
    if query.startswith("/f "):
        clean_query = query[3:].strip()
        return clean_query, fuzzy_search(clean_query, images, top_n=top_n), "fuzzy"
    if query.startswith("/s "):
        clean_query = query[3:].strip()
        return (
            clean_query,
            semantic_search(clean_query, model, embeddings, images, top_n=top_n),
            "semantic",
        )
    return (
        query,
        hybrid_search(query, model, embeddings, images, top_n=top_n),
        "hybrid",
    )


class MyGoEngine:
    """Long-lived search engine shared by the CLI and the macOS app."""

    def __init__(self, status: StatusCallback | None = print):
        self.status = status
        self.images = None
        self.model = None
        self.embeddings = None

    @property
    def is_ready(self):
        return self.model is not None and self.embeddings is not None and self.images is not None

    def load(self):
        self.images = load_images(self.status)
        self.model = load_model(self.status)
        self.embeddings = build_embeddings(
            self.model,
            self.images,
            status=self.status,
            show_progress_bar=self.status is print,
        )
        report(self.status, "MyGo is ready")
        return self

    def search(self, query, top_n=12):
        if not self.is_ready:
            raise RuntimeError("MyGo engine is not loaded")
        query = query.strip()
        if not query:
            return []
        _, results, _ = route_search(
            query,
            self.model,
            self.embeddings,
            self.images,
            top_n=top_n,
        )
        return results


def download_image(url):
    """Download an image with URL-safe path handling."""
    from urllib.parse import quote, urlparse, urlunparse

    parsed = urlparse(url)
    safe_url = urlunparse(parsed._replace(path=quote(parsed.path, safe="/")))
    req = urllib.request.Request(safe_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CTX) as resp:
        return resp.read()


def copy_image_to_clipboard(url):
    """Download image and copy to macOS clipboard"""
    img_data = download_image(url)
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


def fzf_mode(query, model, embeddings, images, top_n=20):
    """Pipe hybrid or explicitly selected search results into fzf."""
    _, results, _ = route_search(
        query,
        model,
        embeddings,
        images,
        top_n=top_n,
    )
    # Format: "idx | score | alt | episode | url"
    lines = []
    for i, (score, img) in enumerate(results):
        lines.append(f"{img['alt']}\t{score:.4f}\t{img['episode']}\t{img['url']}")
    fzf_input = "\n".join(lines)
    preview_script = os.path.join(BASE_DIR, "preview.sh")
    try:
        # Write input to temp file so fzf can use stdin/stdout from tty
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(fzf_input)
            tmp_input = f.name
        proc = subprocess.run(
            f'cat "{tmp_input}" | fzf --delimiter="\t"'
            f' --with-nth=1,2,3'
            f' --header="Select image to copy (TAB fields: alt / score / episode)"'
            f' --preview="{preview_script} {{-1}}"'
            f' --preview-window=right,60%',
            shell=True, capture_output=True, text=True
        )
        os.unlink(tmp_input)
    except FileNotFoundError:
        print("fzf not found, please install it: brew install fzf")
        return
    if proc.returncode != 0:
        return
    selected = proc.stdout.strip()
    if not selected:
        return
    parts = selected.split("\t")
    url = parts[-1]
    alt = parts[0]
    print(f"Copying \"{alt}\"...")
    try:
        copy_image_to_clipboard(url)
        print("Copied to clipboard!")
    except Exception as e:
        print(f"Failed to copy: {e}")


def main():
    use_fzf = "--no-fzf" not in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-fzf"]

    images = load_images()
    model = load_model()
    embeddings = build_embeddings(model, images)
    print()

    if use_fzf:
        if args:
            query = " ".join(args)
            fzf_mode(query, model, embeddings, images)
        else:
            # Loop: search -> fzf -> copy, repeat
            print("fzf mode (q to quit)\n")
            while True:
                try:
                    query = input("Search> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not query or query.lower() == "q":
                    break
                fzf_mode(query, model, embeddings, images)
                print()
        return

    if args:
        query = " ".join(args)
        clean_query, results, mode = route_search(
            query,
            model,
            embeddings,
            images,
        )
        print_results(clean_query, results, mode)
        return

    # Interactive mode
    print("Search MyGo images (hybrid by default, /s semantic, /f fuzzy, q to quit)")
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
        query, last_results, mode = route_search(
            raw,
            model,
            embeddings,
            images,
        )
        print_results(query, last_results, mode)


if __name__ == "__main__":
    main()
