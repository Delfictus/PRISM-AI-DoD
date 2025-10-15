#!/bin/bash
#
# Worker 8 Auto-Sync Daemon
# Automatically syncs changes every 30 minutes
#

WORKER_ID=8
WORKER_DIR="/home/diddy/Desktop/PRISM-Worker-8"
BRANCH="worker-8-finance-deploy"
SYNC_INTERVAL=1800  # 30 minutes
PID_FILE="/tmp/worker-8-autosync.pid"

function start_daemon() {
    if [ -f "$PID_FILE" ]; then
        if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
            echo "⚠️  Auto-sync daemon already running (PID: $(cat $PID_FILE))"
            exit 1
        fi
    fi

    echo "🔄 Starting auto-sync daemon for Worker 8..."

    # Run in background
    nohup bash -c "
        while true; do
            cd '$WORKER_DIR'

            # Check if there are changes to commit
            if ! git diff --quiet || ! git diff --cached --quiet; then
                echo \"\$(date '+%Y-%m-%d %H:%M:%S') - Auto-syncing changes...\"

                # Stage all changes
                git add -A

                # Commit with timestamp
                git commit -m \"autosync: Save progress - \$(date '+%Y-%m-%d %H:%M:%S')

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>\" 2>&1 | tail -3

                # Push to remote
                git push origin $BRANCH 2>&1 | tail -3

                echo \"\$(date '+%Y-%m-%d %H:%M:%S') - Auto-sync complete\"
            fi

            sleep $SYNC_INTERVAL
        done
    " > /tmp/worker-8-autosync.log 2>&1 &

    echo $! > "$PID_FILE"
    echo "✅ Auto-sync daemon started (PID: $(cat $PID_FILE))"
    echo "   Syncing every $((SYNC_INTERVAL / 60)) minutes"
    echo "   Log: /tmp/worker-8-autosync.log"
}

function stop_daemon() {
    if [ ! -f "$PID_FILE" ]; then
        echo "⚠️  Auto-sync daemon not running"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "🛑 Stopping auto-sync daemon (PID: $PID)..."
        kill $PID
        rm "$PID_FILE"
        echo "✅ Auto-sync daemon stopped"
    else
        echo "⚠️  Daemon PID $PID not found - cleaning up"
        rm "$PID_FILE"
    fi
}

function status_daemon() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "✅ Auto-sync daemon running (PID: $PID)"
            echo "   Syncing every $((SYNC_INTERVAL / 60)) minutes"
            echo "   Last 5 log entries:"
            tail -5 /tmp/worker-8-autosync.log
        else
            echo "⚠️  Daemon PID $PID not running - stale PID file"
            rm "$PID_FILE"
        fi
    else
        echo "❌ Auto-sync daemon not running"
    fi
}

case "${1:-status}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        stop_daemon
        sleep 1
        start_daemon
        ;;
    status)
        status_daemon
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
