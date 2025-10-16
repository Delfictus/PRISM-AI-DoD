#!/bin/bash
# PRISM-AI Enhanced Dashboard Launcher with Full Fidelity Metrics
# Provides AlphaFold2-level metrics plus PRISM-AI exclusive features

echo "=============================================="
echo "   PRISM-AI Full Fidelity Metrics Dashboard  "
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python dependencies are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip3 install -r requirements.txt
fi

# Additional dependencies for enhanced metrics
if ! python3 -c "import numpy" 2>/dev/null; then
    echo -e "${YELLOW}Installing NumPy for advanced metrics...${NC}"
    pip3 install numpy
fi

# Build PRISM-AI if needed
if [ ! -f "./target/release/prism-ai" ]; then
    echo -e "${YELLOW}Building PRISM-AI binary with GPU support...${NC}"
    cargo build --release --features cuda
fi

# Start the enhanced API server
echo -e "${GREEN}Starting PRISM-AI Enhanced API server...${NC}"
echo -e "${PURPLE}This server provides:${NC}"
echo "  • Per-residue confidence (pLDDT) scores"
echo "  • Predicted Aligned Error (PAE) matrices"
echo "  • Contact probability maps"
echo "  • Ramachandran plot analysis"
echo "  • Secondary structure predictions"
echo "  • MSA coverage statistics"
echo "  • Domain identification"
echo "  • Quantum coherence metrics (PRISM exclusive)"
echo "  • Thermodynamic free energy (PRISM exclusive)"
echo ""

python3 api_server_enhanced.py &
API_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}✓ Enhanced API server running on http://localhost:8000${NC}"
    echo -e "${GREEN}✓ Full Fidelity Dashboard: file://$(pwd)/dashboard/enhanced.html${NC}"
    echo ""

    echo -e "${PURPLE}Available Metrics:${NC}"
    echo "  Standard AlphaFold2 Metrics:"
    echo "    • Global & per-residue pLDDT"
    echo "    • TM-Score, RMSD, GDT-TS, GDT-HA"
    echo "    • PAE matrix visualization"
    echo "    • Ramachandran statistics"
    echo "    • Secondary structure composition"
    echo "    • MSA depth and coverage"
    echo ""
    echo "  PRISM-AI Exclusive Metrics:"
    echo "    • Quantum coherence scores"
    echo "    • Thermodynamic free energy"
    echo "    • Causal manifold dimensions"
    echo "    • Energy landscape ruggedness"
    echo "    • Active inference confidence"
    echo "    • Real-time GPU utilization"
    echo ""

    # Try to open in browser
    if command -v xdg-open > /dev/null; then
        xdg-open "file://$(pwd)/dashboard/enhanced.html"
    elif command -v open > /dev/null; then
        open "file://$(pwd)/dashboard/enhanced.html"
    else
        echo "Please open dashboard/enhanced.html in your browser"
    fi

    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"

    # Wait for Ctrl+C
    trap "echo -e '\n${YELLOW}Shutting down enhanced server...${NC}'; kill $API_PID; exit" INT
    wait $API_PID
else
    echo -e "${RED}✗ Failed to start enhanced API server${NC}"
    exit 1
fi