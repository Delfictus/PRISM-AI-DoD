#!/bin/bash
# Test API key configuration

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║            PRISM-AI API Key Validation Test                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Load environment
source .env

echo "🔍 Checking API keys..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check OpenAI
if [[ "${OPENAI_API_KEY:0:8}" == "sk-proj-" ]] && [[ ${#OPENAI_API_KEY} -gt 50 ]]; then
    echo "✅ OpenAI: Key configured (${#OPENAI_API_KEY} chars)"
    echo "   Testing connection..."
    curl -s https://api.openai.com/v1/models \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        | grep -q "gpt-4" && echo "   ✅ OpenAI API connection successful!" || echo "   ⚠️  Could not verify connection"
else
    echo "❌ OpenAI: Key not configured or invalid format"
fi

echo ""

# Check Anthropic
if [[ "${ANTHROPIC_API_KEY:0:7}" == "sk-ant-" ]] && [[ ${#ANTHROPIC_API_KEY} -gt 50 ]]; then
    echo "✅ Anthropic: Key configured (${#ANTHROPIC_API_KEY} chars)"
    echo "   Testing connection..."
    response=$(curl -s -o /dev/null -w "%{http_code}" https://api.anthropic.com/v1/messages \
        -H "x-api-key: $ANTHROPIC_API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        -H "content-type: application/json" \
        -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":1,"messages":[{"role":"user","content":"Hi"}]}')
    if [[ "$response" == "200" ]]; then
        echo "   ✅ Anthropic API connection successful!"
    else
        echo "   ⚠️  Connection returned HTTP $response"
    fi
else
    echo "❌ Anthropic: Key not configured or invalid format"
fi

echo ""

# Check Gemini
if [[ -n "$GEMINI_API_KEY" ]] && [[ "$GEMINI_API_KEY" != "..." ]]; then
    echo "✅ Gemini: Key configured (${#GEMINI_API_KEY} chars)"
    echo "   Testing connection..."
    curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY" \
        | grep -q "models" && echo "   ✅ Gemini API connection successful!" || echo "   ⚠️  Could not verify connection (key may be invalid)"
else
    echo "❌ Gemini: Key not configured"
fi

echo ""

# Check Grok
if [[ "${XAI_API_KEY:0:4}" == "xai-" ]] && [[ ${#XAI_API_KEY} -gt 50 ]]; then
    echo "✅ Grok: Key configured (${#XAI_API_KEY} chars)"
    echo "   Testing connection..."
    response=$(curl -s -o /dev/null -w "%{http_code}" https://api.x.ai/v1/chat/completions \
        -H "Authorization: Bearer $XAI_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"grok-2-1212","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}')
    if [[ "$response" == "200" ]]; then
        echo "   ✅ Grok API connection successful!"
    else
        echo "   ⚠️  Connection returned HTTP $response"
    fi
else
    echo "❌ Grok: Key not configured or invalid format"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Summary:"
echo ""

configured=0
[[ "${OPENAI_API_KEY:0:8}" == "sk-proj-" ]] && ((configured++))
[[ "${ANTHROPIC_API_KEY:0:7}" == "sk-ant-" ]] && ((configured++))
[[ -n "$GEMINI_API_KEY" && "$GEMINI_API_KEY" != "..." ]] && ((configured++))
[[ "${XAI_API_KEY:0:4}" == "xai-" ]] && ((configured++))

echo "   Configured providers: $configured/4"
echo ""

if [[ $configured -ge 1 ]]; then
    echo "✅ You have at least one LLM provider configured!"
    echo "   PRISM-AI Mission Charlie can now use LLM intelligence."
else
    echo "❌ No LLM providers configured"
    echo "   Please add at least one API key to .env"
fi

echo ""
echo "Next steps:"
echo "  1. Fix compilation issues (see build errors)"
echo "  2. Run: cargo run --bin prism"
echo "  3. Test LLM integration"
echo ""
