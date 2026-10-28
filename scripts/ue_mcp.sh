#!/usr/bin/env bash
# Run Unreal Editor 5.8 headless with Epic's built-in MCP server, so Claude Code
# can drive the editor over http://127.0.0.1:8127/mcp (see .mcp.json).
#
#   scripts/ue_mcp.sh start|stop|status|log
#
# Port 8127, not the plugin default 8000: 8000 is taken on this machine. The
# listener is bound to loopback via [HTTPServer.Listeners] in the project's
# DefaultEngine.ini — the editor's HTTP server binds to all interfaces otherwise,
# and this endpoint can run arbitrary editor operations.
set -euo pipefail
UE=/opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor
PROJECT=${SURVSIM_UPROJECT:-$HOME/data/ue-projects/SurvSim/SurvSim.uproject}
PORT=${UE_MCP_PORT:-8127}
LOG=${UE_MCP_LOG:-/tmp/survsim_ue_mcp.log}
PIDFILE=/tmp/survsim_ue_mcp.pid

running() { [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case ${1:-status} in
  start)
    if running; then echo "already running (pid $(cat $PIDFILE))"; exit 0; fi
    # setsid so the editor outlives this shell; xvfb-run because there is no
    # display requirement for MCP but the editor still wants one.
    setsid xvfb-run -a -s "-screen 0 1920x1080x24" \
      "$UE" "$PROJECT" -ModelContextProtocolStartServer \
      -ModelContextProtocolPort="$PORT" \
      -ExecCmds="py $(dirname "$(readlink -f "$0")")/ue/job_runner.py" -unattended -nosplash -NoSound \
      >"$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "starting (pid $!), log $LOG; waiting for port $PORT..."
    for _ in $(seq 1 180); do
      if curl -s -o /dev/null "http://127.0.0.1:$PORT/mcp" -X POST \
           -H 'Content-Type: application/json' -d '{}' ; then
        echo "MCP up on http://127.0.0.1:$PORT/mcp; job runner: /tmp/survsim_jobs"; exit 0; fi
      running || { echo "editor exited; see $LOG"; exit 1; }
      sleep 2
    done
    echo "timed out waiting for MCP; see $LOG"; exit 1 ;;
  stop)
    if running; then
      # Kill the whole session (xvfb-run + Xvfb + editor) by process group.
      kill -- -"$(ps -o pgid= "$(cat $PIDFILE)" | tr -d ' ')" 2>/dev/null || true
      rm -f "$PIDFILE"; echo stopped
    else echo "not running"; fi ;;
  status) running && echo "running (pid $(cat $PIDFILE))" || echo "not running" ;;
  log) tail -n 60 "$LOG" ;;
  *) echo "usage: $0 start|stop|status|log"; exit 2 ;;
esac
