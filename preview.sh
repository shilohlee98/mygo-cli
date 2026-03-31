#!/bin/bash
# Preview script for fzf - display image from URL with chafa
# Use symbols mode in tmux (no graphic protocol support), auto-detect otherwise
if [ -n "$TMUX" ]; then
  FORMAT="--format=symbols"
fi
curl -sL --insecure "$1" | chafa $FORMAT --size "${FZF_PREVIEW_COLUMNS:-80}x${FZF_PREVIEW_LINES:-24}" -
