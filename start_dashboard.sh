#!/bin/bash
# PRISM-AI Dashboard Launcher
# Starts the API server and opens the web dashboard

echo "======================================"
echo "   PRISM-AI Competition Dashboard     "
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python dependencies are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip3 install -r requirements.txt
fi

# Build PRISM-AI if needed
if [ ! -f "./target/release/prism-ai" ]; then
    echo -e "${YELLOW}Building PRISM-AI binary...${NC}"
    cargo build --release --features cuda
fi

# Start the API server in the background
echo -e "${GREEN}Starting PRISM-AI API server...${NC}"
python3 api_server.py &
API_PID=$!

# Wait for server to start
sleep 2

# Check if server is running
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}✓ API server running on http://localhost:8000${NC}"
    echo -e "${GREEN}✓ Dashboard available at: file://$(pwd)/dashboard/index.html${NC}"
    echo ""
    echo "Opening dashboard in browser..."

    # Try to open in browser (works on most systems)
    if command -v xdg-open > /dev/null; then
        xdg-open "file://$(pwd)/dashboard/index.html"
    elif command -v open > /dev/null; then
        open "file://$(pwd)/dashboard/index.html"
    else
        echo "Please open dashboard/index.html in your browser"
    fi

    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"

    # Wait for Ctrl+C
    trap "echo -e '\n${YELLOW}Shutting down...${NC}'; kill $API_PID; exit" INT
    wait $API_PID
else
    echo -e "${RED}✗ Failed to start API server${NC}"
    exit 1
fi