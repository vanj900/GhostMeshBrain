#!/usr/bin/env bash
# ghostbrain.sh — GhostMesh central pulse loop
#
# Environment variables (all optional):
#   GHOST_PULSE          Heartbeat interval in seconds (default: 5)
#   GHOST_STATE_FILE     Path to metabolic JSON state (default: /dev/shm/ghost_metabolic.json)
#   GHOST_DIARY_PATH     Path to RAM diary SQLite DB (default: /dev/shm/ghost_diary.db)
#   GHOST_COMPUTE_LOAD   Compute burden per tick (default: 1.0)
#   GHOST_HUD            Show HUD on each tick: 1 = yes, 0 = no (default: 1)
#   GHOST_PYTHON         Python interpreter to use (default: python3)
#   GHOST_LOG_FILE       Optional log file path
#   OLLAMA_URL           Ollama base URL for LLM calls (default: http://localhost:11434)
#   OLLAMA_MODEL         Model name for Ollama (default: mistral)

set -euo pipefail

GHOST_PULSE="${GHOST_PULSE:-5}"
GHOST_STATE_FILE="${GHOST_STATE_FILE:-/dev/shm/ghost_metabolic.json}"
GHOST_DIARY_PATH="${GHOST_DIARY_PATH:-/dev/shm/ghost_diary.db}"
GHOST_COMPUTE_LOAD="${GHOST_COMPUTE_LOAD:-1.0}"
GHOST_HUD="${GHOST_HUD:-1}"
GHOST_PYTHON="${GHOST_PYTHON:-python3}"
GHOST_LOG_FILE="${GHOST_LOG_FILE:-}"

export GHOST_STATE_FILE GHOST_DIARY_PATH GHOST_COMPUTE_LOAD GHOST_HUD
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-mistral}"

# ── Logging helper ──────────────────────────────────────────────────────────
_log() {
    local msg="[$(date '+%Y-%m-%dT%H:%M:%S')] $*"
    echo "$msg"
    if [[ -n "$GHOST_LOG_FILE" ]]; then
        echo "$msg" >> "$GHOST_LOG_FILE"
    fi
}

# ── Ensure /dev/shm is available (fall back to /tmp) ───────────────────────
if [[ ! -d /dev/shm ]]; then
    _log "WARN: /dev/shm not available — falling back to /tmp (non-ephemeral)"
    export GHOST_STATE_FILE="${GHOST_STATE_FILE/\/dev\/shm/\/tmp}"
    export GHOST_DIARY_PATH="${GHOST_DIARY_PATH/\/dev\/shm/\/tmp}"
fi

# ── Module check ─────────────────────────────────────────────────────────────
if ! $GHOST_PYTHON -c "import thermodynamic_agency" 2>/dev/null; then
    _log "ERROR: thermodynamic_agency package not found."
    _log "       Install with: pip install -e ."
    exit 1
fi

_log "GhostMesh pulse starting — interval=${GHOST_PULSE}s"
_log "State file : $GHOST_STATE_FILE"
_log "Diary path : $GHOST_DIARY_PATH"

# ── Main heartbeat ────────────────────────────────────────────────────────────
while true; do
    # Call the metabolic tick — returns one of: FORAGE REST REPAIR DECIDE
    # or DEATH:<ExceptionClass> on fatal error.
    ACTION=$($GHOST_PYTHON -m thermodynamic_agency.core.metabolic 2>/dev/null || echo "TICK_ERROR")

    case "$ACTION" in
        FORAGE)
            _log "↑ FORAGE — energy critical; replenishing resources."
            $GHOST_PYTHON -c "
from thermodynamic_agency.pulse import GhostMesh
m = GhostMesh(); m._forage(); m._save_state()
" 2>/dev/null || _log "WARN: FORAGE action failed"
            ;;

        REST)
            _log "~ REST — high waste/heat; running Janitor."
            $GHOST_PYTHON -c "
from thermodynamic_agency.pulse import GhostMesh
m = GhostMesh(); m._rest(); m._save_state()
" 2>/dev/null || _log "WARN: REST action failed"
            ;;

        REPAIR)
            _log "✦ REPAIR — low integrity/stability; running Surgeon."
            $GHOST_PYTHON -c "
from thermodynamic_agency.pulse import GhostMesh
m = GhostMesh(); m._repair(); m._save_state()
" 2>/dev/null || _log "WARN: REPAIR action failed"
            ;;

        DECIDE)
            _log "● DECIDE — healthy; running active-inference planning."
            $GHOST_PYTHON -c "
from thermodynamic_agency.pulse import GhostMesh
m = GhostMesh(); m._decide(); m._save_state()
" 2>/dev/null || _log "WARN: DECIDE action failed"
            ;;

        DEATH:*)
            CAUSE="${ACTION#DEATH:}"
            _log "☠  DEATH — ${CAUSE}. Organism terminated."
            _log "   Final state: $(cat "$GHOST_STATE_FILE" 2>/dev/null || echo 'unavailable')"
            exit 42
            ;;

        TICK_ERROR)
            _log "WARN: tick produced an error — skipping this pulse."
            ;;

        *)
            _log "WARN: unknown action token '${ACTION}' — skipping."
            ;;
    esac

    # HUD render (separate from tick so we always show current state)
    if [[ "$GHOST_HUD" == "1" && -f "$GHOST_STATE_FILE" ]]; then
        $GHOST_PYTHON -c "
import json, sys
from thermodynamic_agency.interface.hud import print_hud
with open('$GHOST_STATE_FILE') as f:
    state = json.load(f)
print_hud(state)
" 2>/dev/null || true
    fi

    sleep "$GHOST_PULSE"
done
