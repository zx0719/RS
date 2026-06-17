#!/usr/bin/env bash

set -euo pipefail

ROOT="${1:-/mnt/data/mm_data/SAR/dection}"
MODE="${2:-all}"
CPU_COUNT="$(nproc 2>/dev/null || echo 4)"

if [[ -n "${MAX_JOBS:-}" ]]; then
  MAX_JOBS="$MAX_JOBS"
elif (( CPU_COUNT >= 64 )); then
  MAX_JOBS=4
elif (( CPU_COUNT >= 16 )); then
  MAX_JOBS=3
else
  MAX_JOBS=2
fi

LOG_DIR="$ROOT/.extract_logs"
DONE_DIR="$ROOT/.extract_done"
CATALOG_README="${CATALOG_README:-/mnt/data/mm_data/README.md}"
SECTION_BEGIN="<!-- AUTO-SAR-DECTION-DETAILS:BEGIN -->"
SECTION_END="<!-- AUTO-SAR-DECTION-DETAILS:END -->"

task_lines=(
  "OGSOD-1.0|zip|$ROOT/OGSOD-1.0.zip|$ROOT|$ROOT/OGSOD-1.0"
  "RarePlanes-train-psrgb|targz|$ROOT/RarePlanes/train/RarePlanes_train_PS-RGB_tiled.tar.gz|$ROOT/RarePlanes/train|$ROOT/RarePlanes/train/PS-RGB_tiled"
  "MSAR-1.0-main|rar|$ROOT/MSAR-1.0/MSAR-1.0 dataset.rar|$ROOT/MSAR-1.0|$ROOT/MSAR-1.0/MSAR-1.0 dataset"
  "RarePlanes-test-psrgb|targz|$ROOT/RarePlanes/test/RarePlanes_test_PS-RGB_tiled.tar.gz|$ROOT/RarePlanes/test|$ROOT/RarePlanes/test/PS-RGB_tiled"
  "MSAR-1.0-original-images|rar|$ROOT/MSAR-1.0/MSAR-1.0 dataset - large scene original images.rar|$ROOT/MSAR-1.0|$ROOT/MSAR-1.0/MSAR-1.0 dataset - large scene original images"
  "SAR-aircraft|zip|$ROOT/SAR-aircraft.zip|$ROOT|$ROOT/SAR-aircraft"
  "MSAR-1.0-labeled-large-scene|rar|$ROOT/MSAR-1.0/MSAR-1.0 dataset - labeled large scene images.rar|$ROOT/MSAR-1.0|$ROOT/MSAR-1.0/MSAR-1.0 dataset - labeled large scene images"
  "RarePlanes-train-geojson|targz|$ROOT/RarePlanes/train/RarePlanes_train_geojson_aircraft_tiled.tar.gz|$ROOT/RarePlanes/train|$ROOT/RarePlanes/train/geojson_aircraft_tiled"
  "RarePlanes-test-geojson|targz|$ROOT/RarePlanes/test/RarePlanes_test_geojson_aircraft_tiled.tar.gz|$ROOT/RarePlanes/test|$ROOT/RarePlanes/test/geojson_aircraft_tiled"
)

readme_task_lines=(
  "$ROOT/OGSOD-1.0.zip|$ROOT/OGSOD-1.0|zip"
  "$ROOT/SAR-aircraft.zip|$ROOT/SAR-aircraft|zip"
  "$ROOT/RarePlanes/train/RarePlanes_train_PS-RGB_tiled.tar.gz|$ROOT/RarePlanes/train/PS-RGB_tiled|targz"
  "$ROOT/RarePlanes/train/RarePlanes_train_geojson_aircraft_tiled.tar.gz|$ROOT/RarePlanes/train/geojson_aircraft_tiled|targz"
  "$ROOT/RarePlanes/test/RarePlanes_test_PS-RGB_tiled.tar.gz|$ROOT/RarePlanes/test/PS-RGB_tiled|targz"
  "$ROOT/RarePlanes/test/RarePlanes_test_geojson_aircraft_tiled.tar.gz|$ROOT/RarePlanes/test/geojson_aircraft_tiled|targz"
  "$ROOT/MSAR-1.0/MSAR-1.0 dataset.rar|$ROOT/MSAR-1.0/MSAR-1.0 dataset|rar"
  "$ROOT/MSAR-1.0/MSAR-1.0 dataset - labeled large scene images.rar|$ROOT/MSAR-1.0/MSAR-1.0 dataset - labeled large scene images|rar"
  "$ROOT/MSAR-1.0/MSAR-1.0 dataset - large scene original images.rar|$ROOT/MSAR-1.0/MSAR-1.0 dataset - large scene original images|rar"
)

is_nonempty_dir() {
  local path="$1"
  [[ -d "$path" ]] || return 1
  find "$path" -mindepth 1 -print -quit | grep -q .
}

count_files_by_ext() {
  local dir="$1"
  shift
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi

  local -a expr=()
  local first=1
  local ext
  for ext in "$@"; do
    if (( first )); then
      first=0
    else
      expr+=(-o)
    fi
    expr+=(-iname "*.${ext}")
  done

  find "$dir" -type f \( "${expr[@]}" \) | wc -l | tr -d ' '
}

preview_children() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "-"
    return
  fi

  local total
  total="$(find "$dir" -maxdepth 1 -mindepth 1 | wc -l | tr -d ' ')"
  if [[ "$total" == "0" ]]; then
    echo "-"
    return
  fi

  local preview
  preview="$(
    find "$dir" -maxdepth 1 -mindepth 1 -printf '%f\n' \
      | sort \
      | head -n 5 \
      | awk 'BEGIN { first = 1 } { if (!first) printf " / "; printf "%s", $0; first = 0 }'
  )"

  if (( total > 5 )); then
    echo "${preview}, ..."
  else
    echo "$preview"
  fi
}

size_of_path() {
  local path="$1"
  du -sh "$path" 2>/dev/null | awk '{print $1}'
}

write_marker() {
  local marker="$1"
  local archive="$2"
  local output_dir="$3"
  {
    echo "archive=$archive"
    echo "output_dir=$output_dir"
    date -Is
  } > "$marker"
}

extract_one() {
  local task_line="$1"
  local name kind archive out_dir check_dir
  IFS='|' read -r name kind archive out_dir check_dir <<< "$task_line"

  local log_path="$LOG_DIR/${name}.log"
  local marker_path="$DONE_DIR/${name}.done"

  if [[ -f "$marker_path" ]] && is_nonempty_dir "$check_dir"; then
    echo "[$name] skip: marker exists and output is ready"
    return 0
  fi

  if is_nonempty_dir "$check_dir"; then
    if [[ "$kind" == "rar" ]]; then
      echo "[$name] found existing directory without marker; rerunning rar extraction to repair/complete contents"
    else
      echo "[$name] skip: detected existing extracted content in $check_dir"
      write_marker "$marker_path" "$archive" "$check_dir"
      return 0
    fi
  fi

  if [[ ! -f "$archive" ]]; then
    echo "[$name] archive not found: $archive" >&2
    return 1
  fi

  echo "[$name] extracting $archive"

  case "$kind" in
    zip)
      7z x -y -mmt=on -bso1 -bsp1 -o"$out_dir" "$archive" > "$log_path" 2>&1
      ;;
    rar)
      unrar x -o+ "$archive" "$out_dir/" > "$log_path" 2>&1
      ;;
    targz)
      (
        cd "$out_dir"
        tar -I pigz -xf "$archive"
      ) > "$log_path" 2>&1
      ;;
    *)
      echo "[$name] unsupported archive type: $kind" >&2
      return 1
      ;;
  esac

  if ! is_nonempty_dir "$check_dir"; then
    echo "[$name] extraction finished but output looks empty: $check_dir" >&2
    return 1
  fi

  write_marker "$marker_path" "$archive" "$check_dir"
  echo "[$name] done"
}

run_extractions() {
  local failures=0
  local -a batch=()
  local task_line
  local pid

  mkdir -p "$LOG_DIR" "$DONE_DIR"

  for task_line in "${task_lines[@]}"; do
    extract_one "$task_line" &
    batch+=("$!")

    if (( ${#batch[@]} >= MAX_JOBS )); then
      for pid in "${batch[@]}"; do
        if ! wait "$pid"; then
          failures=1
        fi
      done
      batch=()
    fi
  done

  for pid in "${batch[@]}"; do
    if ! wait "$pid"; then
      failures=1
    fi
  done

  if (( failures != 0 )); then
    echo "At least one extraction task failed. Check $LOG_DIR for details." >&2
    exit 1
  fi
}

render_catalog_section() {
  local generated_at
  generated_at="$(date '+%F %T %z')"

  echo "$SECTION_BEGIN"
  echo
  echo "### 10.9 SAR/dection 增补（自动更新）"
  echo
  echo "- Generated at: $generated_at"
  echo "- Dataset root: \`$ROOT\`"
  echo "- 本节补充 2026-04-15 新盘点到的检测子数据集与并行解压状态。"
  echo "- 目录名 \`dection\` 保持磁盘原始拼写，不做重命名。"
  echo
  echo "#### 10.9.1 本次补充解压结果"
  echo
  echo "| Archive | Output Directory | Output Size | Image Files | Annotation Files | Status |"
  echo "| --- | --- | ---: | ---: | ---: | --- |"

  local line archive output_dir kind image_count anno_count status output_size
  for line in "${readme_task_lines[@]}"; do
    IFS='|' read -r archive output_dir kind <<< "$line"
    if is_nonempty_dir "$output_dir"; then
      status="ready"
      output_size="$(size_of_path "$output_dir")"
      image_count="$(count_files_by_ext "$output_dir" jpg jpeg png bmp tif tiff)"
      anno_count="$(count_files_by_ext "$output_dir" xml txt json geojson shp)"
    else
      status="missing"
      output_size="-"
      image_count=0
      anno_count=0
    fi

    echo "| $(basename "$archive") | \`$output_dir\` | $output_size | $image_count | $anno_count | $status |"
  done

  echo
  echo "#### 10.9.2 SAR/dection 顶层目录清单"
  echo
  echo "| Name | Kind | Size | Preview |"
  echo "| --- | --- | ---: | --- |"

  local path base kind size preview
  while IFS= read -r path; do
    base="$(basename "$path")"
    if [[ -d "$path" ]]; then
      kind="directory"
      size="$(size_of_path "$path")"
      preview="$(preview_children "$path")"
    else
      kind="archive"
      size="$(size_of_path "$path")"
      preview="-"
    fi

    echo "| $base | $kind | $size | $preview |"
  done < <(
    find "$ROOT" -maxdepth 1 -mindepth 1 \
      ! -name '.extract_logs' \
      ! -name '.extract_done' \
      ! -name 'README.md' \
      | sort
  )

  echo
  echo "#### 10.9.3 运行说明"
  echo
  echo "- 并行解压脚本：\`bash /home/zhuxiang/RS/SAR/experiments/data/organize_sar_detection.sh \"$ROOT\"\`"
  echo "- 幂等策略：如果目标目录已存在且非空，则直接跳过，不重复覆盖。"
  echo "- 状态目录：\`$ROOT/.extract_logs/\` 和 \`$ROOT/.extract_done/\`。"
  echo
  echo "$SECTION_END"
}

update_catalog_readme() {
  local section_file temp_file
  section_file="$(mktemp)"
  temp_file="$(mktemp)"

  render_catalog_section > "$section_file"

  if grep -Fq "$SECTION_BEGIN" "$CATALOG_README"; then
    awk -v begin="$SECTION_BEGIN" -v end="$SECTION_END" -v insert_file="$section_file" '
      $0 == begin {
        system("cat \"" insert_file "\"")
        skip = 1
        next
      }
      $0 == end {
        skip = 0
        next
      }
      !skip {
        print
      }
    ' "$CATALOG_README" > "$temp_file"
  else
    awk -v marker="## 数据集任务汇总" -v insert_file="$section_file" '
      $0 == marker && !done {
        system("cat \"" insert_file "\"")
        print ""
        done = 1
      }
      {
        print
      }
      END {
        if (!done) {
          print ""
          system("cat \"" insert_file "\"")
        }
      }
    ' "$CATALOG_README" > "$temp_file"
  fi

  mv "$temp_file" "$CATALOG_README"
  rm -f "$section_file"
}

case "$MODE" in
  all)
    run_extractions
    update_catalog_readme
    ;;
  readme-only)
    update_catalog_readme
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    echo "Usage: bash organize_sar_detection.sh [ROOT] [all|readme-only]" >&2
    exit 1
    ;;
esac
