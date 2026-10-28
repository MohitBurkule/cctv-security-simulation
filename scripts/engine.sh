#!/usr/bin/env bash
# Start, stop and health-check the packaged UnrealZoo build.
#
# This exists because two things went wrong repeatedly when driving the engine
# by hand, and both were silent:
#
#   1. `pkill -f unrealzoo` matches the killing command's OWN command line,
#      so the shell killed itself before killing the engine — and on one
#      occasion killed an unrelated Python process instead. Everything here
#      matches on the exact process NAME (`pgrep -x`), never on a pattern.
#
#   2. UnrealCV serves ONE client at a time and auto-increments its port when
#      the configured one is taken. That is how a probe once ended up talking
#      to MinIO's port and later lost its slot to a browser tab. The port is
#      pinned, and `status` checks that the server actually ANSWERS rather than
#      that something is merely listening — a wedged engine holds the socket
#      open and accepts connections while replying to nothing.
#
# Usage: scripts/engine.sh {start|stop|restart|status} [map]

set -uo pipefail

PROC=UnrealZoo_UE5_7
ROOT=${SURVSIM_ENGINE_ROOT:-$HOME/data/unrealzoo/extracted/UnrealZoo_UE5_7_Linux_v3.1.0/Linux/UnrealZoo_UE5_7/Binaries/Linux}
PORT=${SURVSIM_ENGINE_PORT:-9900}
MAP=${2:-/Game/JapanTrainStation/Maps/JapanTrainStation_Optimised}
LOG=${SURVSIM_ENGINE_LOG:-/tmp/survsim-engine.log}
# The venv python, because the health check imports the unrealcv client.
PY_BIN=${SURVSIM_PYTHON:-$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python}

pids() { pgrep -x "$PROC" 2>/dev/null; }

stop() {
  local p; p=$(pids)
  if [ -z "$p" ]; then echo "engine not running"; return 0; fi
  echo "stopping $p"
  kill $p 2>/dev/null
  for _ in $(seq 1 15); do sleep 1; [ -z "$(pids)" ] && break; done
  p=$(pids); [ -n "$p" ] && { echo "forcing"; kill -9 $p 2>/dev/null; sleep 2; }
  echo "stopped"
}

start() {
  [ -n "$(pids)" ] && { echo "already running: $(pids | tr '\n' ' ')"; return 0; }
  [ -x "$ROOT/$PROC" ] || { echo "no engine binary at $ROOT/$PROC" >&2; return 1; }
  echo "starting $PROC on $MAP (port $PORT), log: $LOG"
  ( cd "$ROOT" && setsid ./"$PROC" "$PROC" "$MAP" \
      -RenderOffscreen -nosound -ResX=1280 -ResY=720 >"$LOG" 2>&1 & )
  sleep 3
  echo "pid: $(pids | tr '\n' ' ')"
}

# Health is "does it execute a command", not "is the port open" and not even
# "did it complete the handshake".
#
# Those three are genuinely different states and confusing them cost real time:
# UnrealCV answers its socket handshake from a network thread, but COMMANDS run
# on the game thread. So a build still compiling shaders on first map load
# accepts the connection and completes the handshake while executing nothing —
# which reads as "up" to a handshake check and as "wedged" to a short timeout,
# when the truth is "busy, come back later". A cooked map's first load can take
# many minutes on a laptop GPU.
#
# `status` therefore issues a real command, and reports STARTING rather than
# WEDGED when the process is still burning CPU/GPU — the distinction between
# "needs more time" and "needs a restart".
status() {
  local p; p=$(pids)
  if [ -z "$p" ]; then echo "DOWN — no $PROC process"; return 1; fi
  local pid=${p%% *}
  local cpu; cpu=$(ps -o %cpu= -p "$pid" | tr -d ' ')
  echo "process: $pid  rss: $(ps -o rss= -p "$pid" | tr -d ' ') kB  cpu: ${cpu}%"
  "$PY_BIN" - "$PORT" "$cpu" <<'PY'
import sys
port, cpu = int(sys.argv[1]), float(sys.argv[2] or 0)
try:
    from unrealcv import Client
except ImportError:
    print("cannot check: unrealcv client not importable"); sys.exit(2)
c = Client(("127.0.0.1", port))
c.connect(timeout=10)
if not c.isconnected():
    print(f"DOWN — cannot connect on {port}"); sys.exit(1)
try:
    r = c.request("vget /unrealcv/status", timeout=25)
    if r:
        print(f"UP — executed a command on {port}")
        sys.exit(0)
    # Connected, handshake fine, but the game thread never ran the command.
    state = "STARTING" if cpu > 20 else "WEDGED"
    hint = ("still loading/compiling shaders — wait, do not restart"
            if state == "STARTING" else "idle and unresponsive — restart it")
    print(f"{state} — connected on {port} but no command reply ({hint})")
    sys.exit(1)
finally:
    c.disconnect()
PY
}

# Block until the engine actually executes a command, or give up.
wait_ready() {
  local deadline=${1:-900} waited=0
  while [ "$waited" -lt "$deadline" ]; do
    if status >/dev/null 2>&1; then echo "ready after ${waited}s"; return 0; fi
    sleep 60; waited=$((waited + 60))
  done
  echo "not ready after ${deadline}s"; status; return 1
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  wait) wait_ready "${2:-900}" ;;
  *) echo "usage: $0 {start|stop|restart|status|wait} [map|seconds]" >&2; exit 2 ;;
esac
