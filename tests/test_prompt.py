#!/usr/bin/env python3
"""
Test prompt processing with configurable inference settings.
"""

import aiohttp
import asyncio


async def run_test(base_url: str, test_name: str, payload: dict) -> dict:
    """Run a single test and return results."""
    endpoint = f"{base_url}/v1/chat/completions"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as resp:
                return await resp.json()
    except Exception as e:
        print(f"[{test_name}] Failed: {e}")
        return {"error": str(e)}


async def test_basic_inference(base_url: str = "http://localhost:8080"):
    """Basic inference test."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, what can you do?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    
    result = await run_test("test_basic", payload)
    print(f"Result: {result}")


async def test_chat_history(base_url: str = "http://localhost:8080"):
    """Test with multi-turn chat history."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    
    result = await run_test("chat_history", payload)
    print(f"Result: {result}")


async def test_streaming(base_url: str = "http://localhost:8080"):
    """Test streaming responses."""
    payload = {
        "messages": [
            {"role": "user", "content": "Write a short poem about code."}
        ],
        "stream": True,
        "max_tokens": 50,
    }
    
    result = await run_test("streaming", payload)
    print(f"Result: {result}")


async def test_chat_template(base_url: str = "http://localhost:8080"):
    """Test with various chat templates."""
    test_cases = [
        {
            "name": "simple_chat",
            "payload": {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"}
                ]
            }
        },
        {
            "name": "chat_with_history",
            "payload": {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello! How can I help?"},
                    {"role": "user", "content": "Tell me a joke"}
                ]
            }
        },
        {
            "name": "echo_test",
            "payload": {
                "messages": [
                    {"role": "user", "content": "test echo"}
                ],
                "echo": True,
                "echo_temp": 0.7
            }
        }
    ]
    
    for test in payload:
        result = await run_test(test["name"], {"test_name": test["name"]})
        print(f"Result for {test['name']}: {result}")


async def main():
    """Run all tests."""
    print("Starting prompt processing tests...")
    
    # Run basic inference test
    await run_test("basic_inference", {"message": "Hello"})
    
    # Test chat history handling
    await run_test("chat_history", {"message": "Tell me about Python"}, 
                   previous_messages=[{"role": "assistant", "content": "Sure!"}])
    
    # Test streaming
    await run_test("streaming", {"message": "Count to 10"})
    
    # Test chat templates
    await asyncio.gather(
        run_test("template_basic", {"message": "What is AI?"}),
        run_test("template_with_history", {"message": "Explain more"}, 
                temperature=0.7)
    )
    
    # Test error handling
    await run_test("error_handling", {"message": ""})
    
    print("Tests complete!")


if __name__ == "__main__":
    asyncio.run(run_tests())
