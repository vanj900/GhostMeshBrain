#!/usr/bin/env bash
# ghoststate.sh — One-shot HUD snapshot of the organism's vital signs.
#
# Usage:
#   ./scripts/ghoststate.sh
#   GHOST_STATE_FILE=/tmp/ghost_metabolic.json ./scripts/ghoststate.sh
#
# Prints the full metabolic HUD to stdout and exits.

set -euo pipefail

GHOST_STATE_FILE="${GHOST_STATE_FILE:-/dev/shm/ghost_metabolic.json}"
GHOST_PYTHON="${GHOST_PYTHON:-python3}"

if [[ ! -f "$GHOST_STATE_FILE" ]]; then
    echo "[ghoststate] No state file found at $GHOST_STATE_FILE"
    echo "             Start the pulse loop first: ./scripts/ghostbrain.sh"
    exit 1
fi

$GHOST_PYTHON - <<'PYEOF'
import json, os, sys
from thermodynamic_agency.interface.hud import print_hud

state_file = os.environ.get("GHOST_STATE_FILE", "/dev/shm/ghost_metabolic.json")
try:
    with open(state_file) as f:
        state = json.load(f)
except Exception as e:
    print(f"[ghoststate] Failed to read state: {e}", file=sys.stderr)
    sys.exit(1)

print_hud(state)

# Also print insights if the diary exists
diary_path = os.environ.get("GHOST_DIARY_PATH", "/dev/shm/ghost_diary.db")
if os.path.exists(diary_path):
    try:
        from thermodynamic_agency.memory.diary import RamDiary
        diary = RamDiary(path=diary_path)
        insights = diary.insights()
        entries = diary.recent(5)
        print()
        print("  Recent diary (last 5 entries):")
        for e in entries:
            print(f"    [{e.role:>10}] {e.content[:80]}")
        if insights:
            print()
            print("  Consolidated insights:")
            for ins in insights[-3:]:
                print(f"    • {ins['content'][:100]}")
        diary.close()
    except Exception:
        pass
PYEOF
