#!/usr/bin/env bash
set -euo pipefail

SP="${SP:-/root/.virtualenvs/sglang-0.5.6/lib/python3.10/site-packages}"
DEPS="${DEPS:-/data/my_vllm_test/vllm_020/_remote_deps}"
BACKUP="${SP}/_backup_pre_remote_deps_overlay_$(date +%Y%m%d_%H%M%S)"

OPENAI_SRC="${DEPS}/openai"
OPENAI_DIST="${DEPS}/openai-2.37.0.dist-info"
MHCS_SRC="${DEPS}/model_hosting_container_standards"
MHCS_DIST="${DEPS}/model_hosting_container_standards-0.1.15.dist-info"
JMESPATH_SRC="${DEPS}/jmespath"
JMESPATH_DIST="${DEPS}/jmespath-1.1.0.dist-info"
SUPERVISOR_SRC="${DEPS}/supervisor"
SUPERVISOR_DIST="${DEPS}/supervisor-4.3.0.dist-info"
FLASH_ATTN_IFACE_SRC="${DEPS}/flash_attn_interface.py"
FLASH_ATTN3_SRC="${DEPS}/flash_attn_3"
FLASH_ATTN3_DIST="${DEPS}/flash_attn_3-0.1.4.dist-info"

for path in \
  "${SP}" \
  "${OPENAI_SRC}" "${OPENAI_DIST}" \
  "${MHCS_SRC}" "${MHCS_DIST}" \
  "${JMESPATH_SRC}" "${JMESPATH_DIST}" \
  "${SUPERVISOR_SRC}" "${SUPERVISOR_DIST}" \
  "${FLASH_ATTN_IFACE_SRC}" "${FLASH_ATTN3_SRC}" "${FLASH_ATTN3_DIST}"; do
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

backup_or_remove openai
backup_or_remove model_hosting_container_standards
backup_or_remove jmespath
backup_or_remove supervisor
backup_or_remove flash_attn_interface.py
backup_or_remove flash_attn_3
backup_glob 'openai-*.dist-info'
backup_glob 'model_hosting_container_standards-*.dist-info'
backup_glob 'jmespath-*.dist-info'
backup_glob 'supervisor-*.dist-info'
backup_glob 'flash_attn_3-*.dist-info'

ln -s "${OPENAI_SRC}" "${SP}/openai"
ln -s "${OPENAI_DIST}" "${SP}/$(basename "${OPENAI_DIST}")"
ln -s "${MHCS_SRC}" "${SP}/model_hosting_container_standards"
ln -s "${MHCS_DIST}" "${SP}/$(basename "${MHCS_DIST}")"
ln -s "${JMESPATH_SRC}" "${SP}/jmespath"
ln -s "${JMESPATH_DIST}" "${SP}/$(basename "${JMESPATH_DIST}")"
ln -s "${SUPERVISOR_SRC}" "${SP}/supervisor"
ln -s "${SUPERVISOR_DIST}" "${SP}/$(basename "${SUPERVISOR_DIST}")"
ln -s "${FLASH_ATTN_IFACE_SRC}" "${SP}/flash_attn_interface.py"
ln -s "${FLASH_ATTN3_SRC}" "${SP}/flash_attn_3"
ln -s "${FLASH_ATTN3_DIST}" "${SP}/$(basename "${FLASH_ATTN3_DIST}")"

echo "remote deps overlay installed"
echo "site-packages: ${SP}"
echo "backup: ${BACKUP}"
