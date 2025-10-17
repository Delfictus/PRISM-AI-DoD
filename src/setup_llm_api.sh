#!/bin/bash
# PRISM-AI Mission Charlie - LLM API Setup Script
# This script guides you through setting up LLM API connections

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║        PRISM-AI Mission Charlie - LLM API Setup                   ║"
echo "║        Quantum-Neuromorphic Intelligence Fusion                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if .env exists
if [ -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file already exists!${NC}"
    echo -n "Do you want to overwrite it? (y/N): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Edit .env manually or delete it first."
        exit 0
    fi
fi

# Create .env from template
if [ ! -f .env.example ]; then
    echo -e "${RED}❌ .env.example not found!${NC}"
    echo "Please run this script from the 03-Source-Code directory."
    exit 1
fi

echo ""
echo -e "${BLUE}📋 Creating .env file from template...${NC}"
cp .env.example .env

# Function to prompt for API key
prompt_api_key() {
    local provider=$1
    local env_var=$2
    local url=$3
    local optional=$4

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}$provider${NC}"
    if [ "$optional" = "optional" ]; then
        echo -e "Optional - press Enter to skip"
    else
        echo -e "Required - at least one LLM provider must be configured"
    fi
    echo -e "Get your API key from: ${BLUE}$url${NC}"
    echo ""
    echo -n "Enter your $provider API key (or press Enter to skip): "
    read -r api_key

    if [ -n "$api_key" ]; then
        # Update .env file
        sed -i "s|^$env_var=.*|$env_var=$api_key|" .env
        echo -e "${GREEN}✅ $provider configured${NC}"
        return 0
    else
        echo -e "${YELLOW}⏭️  Skipped $provider${NC}"
        # Comment out the line in .env
        sed -i "s|^$env_var=|# $env_var=|" .env
        return 1
    fi
}

# Prompt for each API key
configured_count=0

if prompt_api_key "OpenAI GPT-4" "OPENAI_API_KEY" "https://platform.openai.com/api-keys" "optional"; then
    ((configured_count++))
fi

if prompt_api_key "Anthropic Claude" "ANTHROPIC_API_KEY" "https://console.anthropic.com/settings/keys" "optional"; then
    ((configured_count++))
fi

if prompt_api_key "Google Gemini" "GEMINI_API_KEY" "https://aistudio.google.com/app/apikey" "optional"; then
    ((configured_count++))
fi

if prompt_api_key "xAI Grok" "XAI_API_KEY" "https://console.x.ai/" "optional"; then
    ((configured_count++))
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if at least one API key was configured
if [ $configured_count -eq 0 ]; then
    echo -e "${RED}❌ No API keys configured!${NC}"
    echo "At least one LLM provider must be configured for Mission Charlie to work."
    echo "Please run this script again and provide at least one API key."
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Configuration complete!${NC}"
echo "Configured $configured_count out of 4 LLM providers."
echo ""

# Optional configuration
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Optional Configuration${NC}"
echo ""

echo -n "Enable GPU acceleration? (Y/n): "
read -r gpu_response
if [[ ! "$gpu_response" =~ ^[Nn]$ ]]; then
    sed -i 's|^MISSION_CHARLIE_ENABLE_GPU=.*|MISSION_CHARLIE_ENABLE_GPU=true|' .env
    echo -e "${GREEN}✅ GPU acceleration enabled${NC}"
else
    sed -i 's|^MISSION_CHARLIE_ENABLE_GPU=.*|MISSION_CHARLIE_ENABLE_GPU=false|' .env
    echo -e "${YELLOW}⏭️  GPU acceleration disabled${NC}"
fi

echo ""
echo -n "Set log level (debug/info/warn/error) [info]: "
read -r log_level
log_level=${log_level:-info}
sed -i "s|^MISSION_CHARLIE_LOG_LEVEL=.*|MISSION_CHARLIE_LOG_LEVEL=$log_level|" .env
echo -e "${GREEN}✅ Log level set to: $log_level${NC}"

echo ""
echo -n "Set semantic cache size (default 10000): "
read -r cache_size
cache_size=${cache_size:-10000}
sed -i "s|^MISSION_CHARLIE_CACHE_SIZE=.*|MISSION_CHARLIE_CACHE_SIZE=$cache_size|" .env
echo -e "${GREEN}✅ Cache size set to: $cache_size${NC}"

# Save configuration
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo "Your configuration has been saved to .env"
echo ""
echo "Next steps:"
echo "  1. Load environment variables:"
echo -e "     ${BLUE}source .env${NC}"
echo ""
echo "  2. Build PRISM-AI:"
echo -e "     ${BLUE}cargo build --release --features mission_charlie${NC}"
echo ""
echo "  3. Run PRISM-AI CLI:"
echo -e "     ${BLUE}cargo run --bin prism${NC}"
echo ""
echo "  4. Or run with environment in one command:"
echo -e "     ${BLUE}env \$(cat .env | xargs) cargo run --bin prism${NC}"
echo ""
echo "Documentation:"
echo "  • Setup guide: LLM_API_SETUP.md"
echo "  • Architecture: API_ARCHITECTURE.md"
echo ""
echo -e "${YELLOW}⚠️  Security reminder:${NC}"
echo "  • .env file contains API keys - never commit to git!"
echo "  • .env is already in .gitignore"
echo "  • Rotate your API keys regularly"
echo ""
echo -e "${GREEN}🚀 PRISM-AI Mission Charlie is ready!${NC}"
