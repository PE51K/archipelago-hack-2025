#!/usr/bin/env bash
# create_submission_archive.sh
# Usage:  ./create_submission_archive.sh [submission-name]
# Example: ./create_submission_archive.sh myproject

set -euo pipefail

# Optional positional argument for the submission name
SUBMISSION_NAME="${1:-}"

# Timestamp in YYYYMMDD-HHMMSS format
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# Build the archive filename
if [[ -n "$SUBMISSION_NAME" ]]; then
    ARCHIVE="submission-${SUBMISSION_NAME}-${TIMESTAMP}.zip"
else
    ARCHIVE="submission-${TIMESTAMP}.zip"
fi

# Patterns to exclude
EXCLUDES=(
  "scripts/*"    "scripts"          # scripts directory
  "examples/*"   "examples"         # examples directory
  "notebooks/*"  "notebooks"        # notebooks directory
  "README.md"    "TODO.md" ".gitignore"  # specific files
)

# Assemble zip command with exclusions
ZIP_ARGS=(-r "$ARCHIVE" .)
for pattern in "${EXCLUDES[@]}"; do
  ZIP_ARGS+=(-x "$pattern")
done

# Create the archive
zip "${ZIP_ARGS[@]}"

echo "Created archive: $ARCHIVE"
