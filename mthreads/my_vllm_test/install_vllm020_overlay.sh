#!/usr/bin/env bash
set -euo pipefail

SP="${SP:-/root/.virtualenvs/sglang-0.5.6/lib/python3.10/site-packages}"
ROOT="${ROOT:-/data/my_vllm_test/vllm_020}"
SRC="${ROOT}/vllm-musa"
DIST="${ROOT}/_dist_info"
BACKUP="${SP}/_backup_pre_vllm020_overlay_$(date +%Y%m%d_%H%M%S)"

VLLM_SRC="${SRC}/third_party/vllm/vllm"
VLLM_MUSA_SRC="${SRC}/vllm_musa"
VLLM_DIST="${DIST}/vllm-0.20.1.dev0+g88d34c640.d20260519.empty.dist-info"
VLLM_MUSA_DIST="${DIST}/vllm_musa-0.1.1.dist-info"

for path in "${SP}" "${VLLM_SRC}" "${VLLM_MUSA_SRC}" "${VLLM_DIST}" "${VLLM_MUSA_DIST}"; do
  if [ ! -e "${path}" ]; then
    echo "missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${BACKUP}"

backup_or_remove() {
  local name="$1"
  local path="${SP}/${name}"
  if [ -L "${path}" ]; then
    unlink "${path}"
  elif [ -e "${path}" ]; then
    mv "${path}" "${BACKUP}/${name}"
  fi
}

backup_glob() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( "${SP}"/${pattern} )
  shopt -u nullglob
  for path in "${matches[@]}"; do
    local name
    name="$(basename "${path}")"
    if [ -L "${path}" ]; then
      unlink "${path}"
    elif [ -e "${path}" ]; then
      mv "${path}" "${BACKUP}/${name}"
    fi
  done
}

backup_or_remove vllm
backup_or_remove vllm_musa
backup_glob 'vllm-*.dist-info'
backup_glob 'vllm_musa-*.dist-info'

ln -s "${VLLM_SRC}" "${SP}/vllm"
ln -s "${VLLM_MUSA_SRC}" "${SP}/vllm_musa"
ln -s "${VLLM_DIST}" "${SP}/$(basename "${VLLM_DIST}")"
ln -s "${VLLM_MUSA_DIST}" "${SP}/$(basename "${VLLM_MUSA_DIST}")"

echo "overlay installed"
echo "site-packages: ${SP}"
echo "backup: ${BACKUP}"
