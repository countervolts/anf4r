#!/usr/bin/env bash
# Copies the system Vulkan headers to build/vulkan-include for the MinGW (Windows DLL) build.
set -euo pipefail
cd "$(dirname "$0")/.."
src="${VULKAN_INCLUDE:-/usr/include}"
[[ -f "$src/vulkan/vulkan.h" ]] || { echo "vulkan headers not found in $src (set VULKAN_INCLUDE)" >&2; exit 1; }
rm -rf build/vulkan-include && mkdir -p build/vulkan-include
cp -r "$src/vulkan" build/vulkan-include/ && { [[ -d "$src/vk_video" ]] && cp -r "$src/vk_video" build/vulkan-include/ || true; }
echo "copied to build/vulkan-include"
