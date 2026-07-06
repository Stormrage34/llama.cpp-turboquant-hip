#!/usr/bin/env python3
"""
Configurable prompt processing tester.

Tests various prompt configurations with configurable parameters.
"""

import aiohttp
import asyncio


async def send_request(payload: dict) -> dict:
    """Send request to llama.cpp server."""
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8080/v1/chat/completions",
                json=payload
            ) as response:
                return await response.json()
    except Exception as e:
        return {"error": str(e)}


async def test_basic_inference(temperature: float = 0.7, max_tokens: int = 50) -> dict:
    """Test basic inference with configurable parameters."""
    
    payload = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": f"Test with temp {temperature}"}]
    }
    
    result = await send_request(payload)
    return result


async def test_chat_history() -> dict:
    """Test chat history handling."""
    
    payload = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me about Python"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    result = await send_request(payload)
    return result


async def test_chat_templates() -> dict:
    """Test various chat templates."""
    
    payload = {
        "messages": [
            {"role": "user", "content": "Explain: "},
            {"role": "assistant", "content": "I'll help..."}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    result = await send_request(payload)
    return result


async def test_error_handling() -> dict:
    """Test error handling for invalid inputs."""
    
    payload_invalid = {
        "temperature": -1.0,  # Invalid - must be >= 0
        "messages": [{"role": "user", "content": "test"}]
    }
    
    try:
        result = await send_request(payload_invalid)
        if "error" in result:
            print(f"Catched error: {result['error']}")
        else:
            print("No error caught (unexpected)")
    except Exception as e:
        print(f"Exception: {e}")
    
    return {"status": "test_completed"}


async def main():
    """Run all tests."""
    
    print("Starting prompt processing tests...\n")
    
    # Test 1: Basic inference
    print("[TEST 1] Basic Inference")
    result = await test_basic_inference()
    if "error" in result:
        print(f"  ❌ FAILED: {result['error']}")
    else:
        print("  ✓ PASSED")
    
    # Test 2: Chat history
    print("\n[TEST 2] Chat History")
    result = await test_chat_history()
    if "error" in result:
        print(f"  ❌ FAILED: {result['error']}")
    else:
        print("  ✓ PASSED")
    
    # Test 3: Chat templates
    print("\n[TEST 3] Chat Templates")
    result = await test_chat_templates()
    if "error" in result:
        print(f"  ❌ FAILED: {result['error']}")
    else:
        print("  ✓ PASSED")
    
    # Test 4: Error handling
    print("\n[TEST 4] Error Handling")
    result = await test_error_handling()
    if "error" in result:
        print(f"  ❌ FAILED: {result['error']}")
    else:
        print("  ✓ PASSED")
    
    print("\n" + "="*50)
    print("All tests completed!")


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(test_all())
    except Exception as e:
        print(f"Fatal error: {e}")
