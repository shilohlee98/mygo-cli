#!/bin/bash
# Preview script for fzf - display image from URL with chafa
# Force symbols mode for tmux compatibility
curl -sL --insecure "$1" | chafa --format=symbols --size "${FZF_PREVIEW_COLUMNS:-80}x${FZF_PREVIEW_LINES:-24}" -
