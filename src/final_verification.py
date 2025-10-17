#!/usr/bin/env python3
import os
import requests
import json
from datetime import datetime

# Load .env
with open('.env', 'r') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

print("=" * 70)
print("FINAL VERIFICATION - COMPLETE TRANSPARENCY CHECK")
print("=" * 70)
print()

# Test each API with actual request and show EVERYTHING

def test_api(name, url, headers, payload):
    print(f"\n{name}:")
    print("-" * 40)
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # Show actual response
            if 'choices' in data:  # OpenAI/Grok format
                content = data['choices'][0]['message']['content']
                model = data.get('model', 'unknown')
                tokens = data.get('usage', {}).get('total_tokens', 'N/A')
                print(f"✅ SUCCESS")
                print(f"   Response: '{content}'")
                print(f"   Model used: {model}")
                print(f"   Tokens used: {tokens}")
                return True
            elif 'content' in data:  # Claude format
                content = data['content'][0]['text']
                model = data.get('model', 'unknown')
                tokens = data.get('usage', {}).get('output_tokens', 'N/A')
                print(f"✅ SUCCESS")
                print(f"   Response: '{content}'")
                print(f"   Model used: {model}")
                print(f"   Tokens used: {tokens}")
                return True
            elif 'candidates' in data:  # Gemini format
                content = data['candidates'][0]['content']['parts'][0]['text']
                print(f"✅ SUCCESS")
                print(f"   Response: '{content}'")
                print(f"   Model used: gemini-2.0-flash-exp")
                return True
            else:
                print(f"⚠️ Unexpected response format")
                print(f"   Raw: {str(data)[:200]}")
                return False
        else:
            print(f"❌ FAILED - HTTP {response.status_code}")
            error_data = response.json() if response.text else {}
            if 'error' in error_data:
                print(f"   Error: {error_data['error'].get('message', 'Unknown')}")
            else:
                print(f"   Raw: {response.text[:200]}")
            return False
    except requests.Timeout:
        print(f"❌ TIMEOUT - Request took longer than 10 seconds")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False

# OpenAI
openai_key = os.environ.get('OPENAI_API_KEY', '')
openai_works = False
if openai_key:
    openai_works = test_api(
        "OpenAI GPT-4",
        "https://api.openai.com/v1/chat/completions",
        {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        },
        {
            "model": "gpt-4-0613",
            "messages": [{"role": "user", "content": "Respond with exactly: WORKING"}],
            "max_tokens": 10,
            "temperature": 0
        }
    )

# Claude
claude_key = os.environ.get('ANTHROPIC_API_KEY', '')
claude_works = False
if claude_key:
    claude_works = test_api(
        "Anthropic Claude",
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": claude_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        },
        {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Respond with exactly: WORKING"}],
            "max_tokens": 10
        }
    )

# Gemini
gemini_key = os.environ.get('GEMINI_API_KEY', '')
gemini_works = False
if gemini_key and gemini_key != '...':
    gemini_works = test_api(
        "Google Gemini",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={gemini_key}",
        {"Content-Type": "application/json"},
        {
            "contents": [{"parts": [{"text": "Respond with exactly: WORKING"}]}],
            "generationConfig": {"maxOutputTokens": 10, "temperature": 0}
        }
    )

# Grok
grok_key = os.environ.get('XAI_API_KEY', '')
grok_works = False
if grok_key:
    grok_works = test_api(
        "xAI Grok",
        "https://api.x.ai/v1/chat/completions",
        {
            "Authorization": f"Bearer {grok_key}",
            "Content-Type": "application/json"
        },
        {
            "model": "grok-2-1212",
            "messages": [{"role": "user", "content": "Respond with exactly: WORKING"}],
            "max_tokens": 10,
            "temperature": 0
        }
    )

# Final summary
print("\n" + "=" * 70)
print("ABSOLUTE TRUTH - FINAL STATUS:")
print("=" * 70)
print()

working = []
not_working = []

if openai_works: working.append("OpenAI GPT-4")
else: not_working.append("OpenAI GPT-4")

if claude_works: working.append("Anthropic Claude")
else: not_working.append("Anthropic Claude")

if gemini_works: working.append("Google Gemini")
else: not_working.append("Google Gemini")

if grok_works: working.append("xAI Grok")
else: not_working.append("xAI Grok")

print(f"✅ WORKING: {len(working)}/4")
for provider in working:
    print(f"   • {provider}")

if not_working:
    print(f"\n❌ NOT WORKING: {len(not_working)}/4")
    for provider in not_working:
        print(f"   • {provider}")

print()
print("This is the complete, unfiltered status.")
print("Nothing hidden, nothing omitted.")
print()

if len(working) >= 3:
    print("VERDICT: System is fully operational with {} providers.".format(len(working)))
elif len(working) >= 2:
    print("VERDICT: System is functional with {} providers (minimum for consensus).".format(len(working)))
elif len(working) == 1:
    print("VERDICT: Limited functionality with only 1 provider.")
else:
    print("VERDICT: No providers working - check keys and connectivity.")
