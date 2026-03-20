#!/bin/bash
# ============================================================
# AskAra+ — Entrypoint for Railway / Render / Docker
# ============================================================

echo "================================================"
echo "  AskAra+ Backend — Starting..."
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================"

cd /app/backend

# ── 1. Start FastMCP server (background) ──
echo "[1/2] Starting FastMCP server on port 8001..."
python mcp_server.py > /tmp/mcp.log 2>&1 &
MCP_PID=$!

# Wait for MCP to initialize
sleep 5

if kill -0 $MCP_PID 2>/dev/null; then
    echo "  ✓ FastMCP running (PID $MCP_PID)"
else
    echo "  ⚠ FastMCP failed to start. MCP log:"
    cat /tmp/mcp.log 2>/dev/null
    echo ""
    echo "  Continuing anyway — FastAPI will start but agent MCP calls may fail."
fi

# ── 2. Start FastAPI server (foreground) ──
FASTAPI_PORT=${PORT:-8000}
echo "[2/2] Starting FastAPI on port $FASTAPI_PORT..."
echo "================================================"

exec uvicorn server:app \
    --host 0.0.0.0 \
    --port "$FASTAPI_PORT" \
    --workers 1 \
    --timeout-keep-alive 120 \
    --log-level info