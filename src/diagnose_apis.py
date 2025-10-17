#!/usr/bin/env python3
import os
import requests
import json

# Load .env file
with open('.env', 'r') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

# Get API keys
openai_key = os.environ.get('OPENAI_API_KEY', '')
anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║              API Diagnosis                                         ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# Test OpenAI
print("1. OpenAI Diagnosis:")
print(f"   Key length: {len(openai_key)} chars")
print(f"   Key prefix: {openai_key[:10] if openai_key else 'None'}")
print(f"   Key suffix: {openai_key[-10:] if openai_key else 'None'}")

if openai_key:
    # First, list available models
    print("\n   Testing model access...")
    response = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {openai_key}"}
    )

    if response.status_code == 200:
        models = response.json()
        available_models = [m['id'] for m in models['data'] if 'gpt' in m['id'].lower()]
        print(f"   ✅ API key valid! Available GPT models: {', '.join(available_models[:5])}")

        # Try completion with first available model
        if available_models:
            model = available_models[0]
            print(f"\n   Testing completion with {model}...")

            completion_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 10
                }
            )

            if completion_response.status_code == 200:
                result = completion_response.json()
                message = result['choices'][0]['message']['content']
                print(f"   ✅ Completion successful: \"{message}\"")
            else:
                error = completion_response.json().get('error', {})
                print(f"   ❌ Completion failed: {error.get('message', 'Unknown error')}")
    else:
        print(f"   ❌ API key invalid or no access. Status: {response.status_code}")
        if response.status_code == 401:
            print("   Issue: Invalid API key")
        elif response.status_code == 429:
            print("   Issue: Rate limited or quota exceeded")

print("\n" + "="*70 + "\n")

# Test Anthropic
print("2. Anthropic Claude Diagnosis:")
print(f"   Key length: {len(anthropic_key)} chars")
print(f"   Key prefix: {anthropic_key[:10] if anthropic_key else 'None'}")
print(f"   Key suffix: {anthropic_key[-10:] if anthropic_key else 'None'}")

if anthropic_key:
    print("\n   Testing Claude API...")

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        },
        json={
            "model": "claude-3-5-sonnet-20241022",  # Try older version
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10
        }
    )

    if response.status_code == 200:
        result = response.json()
        message = result['content'][0]['text']
        print(f"   ✅ Claude working: \"{message}\"")
    else:
        print(f"   ❌ Claude API error. Status: {response.status_code}")
        error_data = response.json() if response.text else {}
        error_msg = error_data.get('error', {}).get('message', response.text[:200])
        print(f"   Error: {error_msg}")

        # Try different model
        print("\n   Trying claude-3-haiku-20240307...")
        response2 = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 10
            }
        )

        if response2.status_code == 200:
            result = response2.json()
            message = result['content'][0]['text']
            print(f"   ✅ Claude Haiku working: \"{message}\"")
        else:
            print(f"   ❌ Claude Haiku also failed: {response2.status_code}")

print("\n" + "="*70 + "\n")
print("Diagnosis complete!")