#!/bin/bash
set -u
src=$1; dst=$2; interval=${3:-45}
mkdir -p "$dst"
echo "MIRROR_START host=$(hostname) src=$src dst=$dst dev_src=$(stat -c %d "$src" 2>/dev/null || echo NA) dev_dst=$(stat -c %d "$dst")"
while true; do
  linked=0; seen=0
  if [ -d "$src" ]; then
    while IFS= read -r -d '' f; do
      seen=$((seen+1)); b=${f##*/}; d=$dst/$b
      if [ ! -e "$d" ]; then
        if ln "$f" "$d" 2>/dev/null; then linked=$((linked+1)); fi
      fi
    done < <(find "$src" -maxdepth 1 -type f -name '*.u16' -print0 2>/dev/null)
  fi
  total=$(find "$dst" -maxdepth 1 -type f -name '*.u16' 2>/dev/null | wc -l)
  echo "MIRROR host=$(hostname) seen=$seen linked=$linked persistent=$total time=$(date +%s)"
  sleep "$interval"
done
