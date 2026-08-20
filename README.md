# mygo-cli

A MyGo meme image search tool with hybrid, semantic, and fuzzy search. Browse results interactively with fzf and copy images to clipboard.

It also includes a native macOS menu bar app. The app keeps the semantic model loaded in the background, so follow-up searches do not pay the model startup cost.


https://github.com/user-attachments/assets/626674c4-4817-47f3-8c6d-4494a7e3bb20

## Download the macOS app

Download the latest Apple Silicon DMG from [GitHub Releases](https://github.com/shilohlee98/mygo-cli/releases), open it, and drag **MyGo.app** into **Applications**. The prebuilt app requires macOS 13 or later.


## Requirements

- Apple Silicon Mac with macOS 13 or later for the prebuilt app
- Python 3.11 and [uv](https://docs.astral.sh/uv/) for CLI/source development
- `fzf` and `chafa` for the optional interactive terminal interface

> **Note:** macOS built-in Terminal.app and tmux render fzf preview poorly. For best experience, use [iTerm2](https://iterm2.com/), [Kitty](https://sw.kovidgoyal.net/kitty/), or [Ghostty](https://ghostty.org/) without tmux.

## External Dependencies

```bash
brew install fzf chafa
```

| Tool | Purpose |
|------|---------|
| [fzf](https://github.com/junegunn/fzf) | Interactive fuzzy finder |
| [chafa](https://github.com/hpjansson/chafa) | Terminal image preview |

## Installation

```bash
git clone https://github.com/shilohlee98/mygo-cli.git
cd mygo-cli
uv tool install .
```

If the tool is already installed and you want the new ONNX dependencies, reinstall it:

```bash
uv tool install --reinstall .
```

On first run, image data is fetched from the API and cached locally. MyGo downloads the prebuilt int8 `multilingual-e5-small` ONNX model and reuses it on later searches. Inference runs directly through ONNX Runtime and does not load PyTorch.

## Usage

After installation, the `mygo` command is available globally:

```bash
# Interactive fzf mode (default)
mygo

# Search with a query directly
mygo haha

# Plain text output (no fzf)
mygo --no-fzf

# Plain text with query
mygo --no-fzf haha
```

Alternatively, run without installing globally:

```bash
uv run mygo
```

Performance toggles:

```bash
# Skip int8 quantization (uses substantially more memory)
MYGO_ONNX_QUANTIZATION=off mygo
```

## macOS menu bar app

Run the app from source while developing:

```bash
uv run mygo-app
```

MyGo opens a native search window, stays visible in the Dock, and keeps running after the window is closed. Normal searches blend multilingual E5 semantic relevance with exact and fuzzy text matching. Prefix the query with `/s ` for semantic-only results or `/f ` for fuzzy-only results. Click **Copy Image** on any result to put it on the macOS clipboard.

Short English greetings are normalized only for semantic retrieval (`hi` and `hello` → `妳好`). Hybrid text scoring still uses the original query, with English word boundaries so `hi` does not match inside names such as `CRYCHIC`.

Build a standalone app bundle:

```bash
./scripts/build_app.sh
open dist/MyGo.app
```

Build an installable DMG:

```bash
./scripts/build_dmg.sh
```

The outputs are `dist/MyGo.app`, `dist/MyGo-<version>-<arch>.dmg`, and a SHA-256 checksum. Build artifacts are intentionally ignored by Git. Local builds use an ad-hoc signature and are suitable for development on your own Mac.

For personal use, move the app to `/Applications`. To start it automatically after login, use **MyGO → Open Login Items Settings…** and add `MyGo.app` in macOS System Settings.

The image catalog, ONNX model, and embeddings are stored under:

```text
~/Library/Application Support/MyGo/
```

The first launch needs network access and takes longer while these files are downloaded and generated. Later launches reuse them. If Application Support is not writable in the current launch context, MyGo reports that it is using a safe temporary cache instead of failing during ONNX setup.

### Interactive Commands (--no-fzf mode)

- Type text — hybrid semantic and text search
- `/s <query>` — semantic-only search
- `/f <query>` — fuzzy search
- Enter a number — copy that image to clipboard
- `q` — quit

## Acknowledgements

Image data provided by [Miyago9267](https://github.com/miyago9267)'s [MyGO-Searcher](https://github.com/miyago9267/MyGO-Searcher) API.
