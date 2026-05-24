#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Resolve project root (using git if available, falling back to physical script location to handle symlinks)
if git rev-parse --show-toplevel >/dev/null 2>&1; then
  PROJECT_ROOT="$(git rev-parse --show-toplevel)"
else
  # Fallback to physical script location (resolving symlinks recursively)
  SOURCE="${BASH_SOURCE[0]}"
  while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
  done
  SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

GITIGNORE="$PROJECT_ROOT/.gitignore"
DOCKERIGNORE="$PROJECT_ROOT/.dockerignore"

# Markers in .dockerignore
START_MARKER="# --- BEGIN GITIGNORE ---"
END_MARKER="# --- END GITIGNORE ---"

# Ensure both files exist
if [ ! -f "$GITIGNORE" ] || [ ! -f "$DOCKERIGNORE" ]; then
  echo "Error: .gitignore or .dockerignore not found in $PROJECT_ROOT"
  exit 1
fi

# Ensure markers exist in .dockerignore; if not, append them
if ! grep -q "$START_MARKER" "$DOCKERIGNORE"; then
  echo -e "\n$START_MARKER\n$END_MARKER" >> "$DOCKERIGNORE"
fi

echo "Syncing .gitignore content into .dockerignore..."

# Create a temporary file
TEMP_FILE=$(mktemp)

# 1. Copy everything up to and including the START_MARKER
sed -n "1,/$START_MARKER/p" "$DOCKERIGNORE" > "$TEMP_FILE"

# 2. Append comment and the contents of .gitignore
echo "# (Auto-generated from .gitignore. Do not edit this section directly.)" >> "$TEMP_FILE"
cat "$GITIGNORE" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

# 3. Copy the END_MARKER and everything after it
sed -n "/$END_MARKER/,\$p" "$DOCKERIGNORE" >> "$TEMP_FILE"

# 4. Overwrite original .dockerignore
mv "$TEMP_FILE" "$DOCKERIGNORE"

# 5. Git add the updated .dockerignore (only if we are inside a Git commit/stage context)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add "$DOCKERIGNORE"
fi

echo "Successfully synchronized .dockerignore!"
