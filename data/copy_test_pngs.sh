#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_SPLIT_CSV="${1:-"${DATA_DIR}/h5s/master_split.csv"}"
PNG_DIR="${2:-"${DATA_DIR}/png"}"
OUTPUT_DIR="${3:-"${DATA_DIR}/test_images"}"
TEST_SPLIT_ID="${4:-2}"

mkdir -p "${OUTPUT_DIR}"

copied=0
missing=0

while IFS=, read -r image_name mask_name split_id extra; do
    split_id="${split_id//$'\r'/}"
    if [[ "${split_id}" != "${TEST_SPLIT_ID}" ]]; then
        continue
    fi

    for filename in "${image_name}" "${mask_name}"; do
        source_path="${PNG_DIR}/${filename}"
        target_path="${OUTPUT_DIR}/${filename}"

        if [[ -f "${source_path}" ]]; then
            cp "${source_path}" "${target_path}"
            copied=$((copied + 1))
        else
            echo "Missing: ${source_path}" >&2
            missing=$((missing + 1))
        fi
    done
done < "${MASTER_SPLIT_CSV}"

echo "Copied ${copied} PNG files to ${OUTPUT_DIR}"
if [[ "${missing}" -gt 0 ]]; then
    echo "Missing ${missing} PNG files" >&2
    exit 1
fi
