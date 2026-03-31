# mygobot

A MyGo meme image search tool with semantic and fuzzy search. Browse results interactively with fzf and copy images to clipboard.

## Requirements

- macOS (clipboard uses `osascript`)
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)

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
git clone <repo-url>
cd mygobot
uv sync
```

On first run, image data is fetched from the API and cached locally. The `paraphrase-multilingual-MiniLM-L12-v2` model is also downloaded for semantic search.

## Usage

```bash
# Interactive fzf mode (default)
uv run mygo

# Search with a query directly
uv run mygo haha

# Plain text output (no fzf)
uv run mygo --no-fzf

# Plain text with query
uv run mygo --no-fzf haha
```

### Interactive Commands (--no-fzf mode)

- Type text — semantic search
- `/f <query>` — fuzzy search
- Enter a number — copy that image to clipboard
- `q` — quit
