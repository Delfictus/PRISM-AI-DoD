#!/bin/bash
source .env

echo "Testing OpenAI API..."
echo "Key length: ${#OPENAI_API_KEY} chars"

# Test with curl directly
response=$(curl -s -X POST https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Reply with exactly: Hello"}],
    "max_tokens": 10,
    "temperature": 0
  }')

echo "Raw response:"
echo "$response" | head -500

# Check for error
if echo "$response" | grep -q '"error"'; then
    echo ""
    echo "❌ Error detected in response"
    error_msg=$(echo "$response" | jq -r '.error.message' 2>/dev/null || echo "Could not parse error")
    echo "Error: $error_msg"
else
    # Try to extract the message
    message=$(echo "$response" | jq -r '.choices[0].message.content' 2>/dev/null || echo "Could not parse response")
    echo ""
    echo "✅ Success! Response: $message"
fi