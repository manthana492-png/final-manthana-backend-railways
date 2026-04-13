#!/usr/bin/env python3
"""Test script to verify the Modal tuple unpacking fix is working."""

import sys
import os
sys.path.insert(0, "/app/shared")
sys.path.insert(0, "/app")

try:
    from llm_router import llm_router, safe_chat_complete_sync
    print("SUCCESS: llm_router imported with safe_chat_complete_sync")
    
    # Test the version check
    try:
        import inspect
        from manthana_inference import chat_complete_sync
        source = inspect.getsource(chat_complete_sync)
        if "return text, model_eff, usage_info" in source:
            print("SUCCESS: manthana-inference has correct 3-tuple return format")
        else:
            print("WARNING: manthana-inference may have outdated format")
    except Exception as e:
        print(f"WARNING: Could not verify manthana-inference: {e}")
    
    # Test a simple completion (this will test the tuple unpacking)
    try:
        print("Testing OpenRouter completion...")
        result = llm_router.complete_for_role(
            "narrative_default",
            "You are a radiologist.",
            "Test message to verify the fix is working.",
            max_tokens=100,
        )
        print(f"SUCCESS: OpenRouter completion worked: {result.get('model_used', 'unknown')}")
        print(f"Content preview: {result.get('content', '')[:100]}...")
        
    except Exception as e:
        print(f"FAILED: OpenRouter completion failed: {e}")
        
except ImportError as e:
    print(f"FAILED: Import error: {e}")
except Exception as e:
    print(f"FAILED: Unexpected error: {e}")

print("Test completed.")
