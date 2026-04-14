#!/bin/bash

# Configuration
ECOSYSTEM_FILE="$PWD/ecosystem.config.js"
APP_NAME=$(grep -m 1 "name:" $ECOSYSTEM_FILE | sed -E "s/.*name: *['\"]([^'\"]+)['\"].*/\1/")
PORT=8000

start_server() {
    # Check if already running
    if pm2 describe "$APP_NAME" > /dev/null 2>&1; then
        STATUS=$(pm2 jlist | jq -r ".[] | select(.name==\"$APP_NAME\") | .pm2_env.status")
        if [ "$STATUS" = "online" ]; then
            echo "${APP_NAME} server already running"
            pm2 describe "$APP_NAME"
            return 1
        fi
    fi

    echo "Starting ${APP_NAME} server with PM2..."
    pm2 start "$ECOSYSTEM_FILE"
    echo "${APP_NAME} Server started (Port: $PORT)"
    echo "Using GPU 0,1,2,3,4,5,6,7 with TP=8 + EP"
    echo "Note: Tool/reasoning parsers require vLLM nightly build"
}

stop_server() {
    if ! pm2 describe "$APP_NAME" > /dev/null 2>&1; then
        echo "${APP_NAME} server is not running"
        return 1
    fi

    echo "Stopping ${APP_NAME} server..."
    pm2 stop "$APP_NAME"
    echo "${APP_NAME} server stopped"
}

restart_server() {
    echo "Restarting ${APP_NAME} server..."
    pm2 restart "$APP_NAME" --update-env
    echo "${APP_NAME} server restarted"
}

reload_server() {
    echo "Reloading ${APP_NAME} server..."
    pm2 reload "$APP_NAME" --update-env
    echo "${APP_NAME} server reloaded"
}

status_server() {
    if pm2 describe "$APP_NAME" > /dev/null 2>&1; then
        pm2 describe "$APP_NAME"
    else
        echo "${APP_NAME} server not found in PM2"
    fi
}

logs_server() {
    pm2 logs "$APP_NAME" --lines 100
}

logs_follow() {
    pm2 logs "$APP_NAME"
}

delete_server() {
    echo "Removing ${APP_NAME} server from PM2..."
    pm2 delete "$APP_NAME"
    echo "${APP_NAME} server removed from PM2"
}

save_config() {
    echo "Saving PM2 process list..."
    pm2 save
    echo "PM2 process list saved (will auto-start on reboot if pm2 startup is configured)"
}

case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 1
        delete_server
        sleep 1
        start_server
        ;;
    reload)
        reload_server
        ;;
    status)
        status_server
        ;;
    logs)
        logs_server
        ;;
    logs-follow)
        logs_follow
        ;;
    delete)
        delete_server
        ;;
    save)
        save_config
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|reload|status|logs|logs-follow|delete|save}"
        echo ""
        echo "Commands:"
        echo "  start       - Start the ${APP_NAME} server"
        echo "  stop        - Stop the server (keeps in PM2 list)"
        echo "  restart     - Restart the server"
        echo "  reload      - Graceful reload"
        echo "  status      - Show server status"
        echo "  logs        - Show last 100 lines of logs"
        echo "  logs-follow - Follow logs in real-time"
        echo "  delete      - Remove from PM2 completely"
        echo "  save        - Save PM2 process list for auto-restart"
        exit 1
        ;;
esac
