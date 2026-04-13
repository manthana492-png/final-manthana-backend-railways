#!/usr/bin/env python3
"""Verify the tuple unpacking fix is working by testing the core logic."""

import sys
import os

# Add paths to match Modal environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
sys.path.insert(0, os.path.dirname(__file__))

print("=== Verifying Tuple Unpacking Fix ===")

# Test 1: Check if our safe_chat_complete_sync function exists
try:
    from shared.llm_router import safe_chat_complete_sync, _safe_unpack_chat_result
    print("SUCCESS: safe_chat_complete_sync function found")
except ImportError as e:
    print(f"FAILED: Import error: {e}")
    sys.exit(1)

# Test 2: Test the _safe_unpack_chat_result function with different tuple formats
print("\nTesting _safe_unpack_chat_result function:")

# Test with 3-tuple (correct format)
try:
    text, model, usage = _safe_unpack_chat_result(("Hello world", "gpt-4", {"tokens": 10}))
    print("SUCCESS: 3-tuple unpacking works")
    print(f"  Text: {text}, Model: {model}, Usage: {usage}")
except Exception as e:
    print(f"FAILED: 3-tuple unpacking failed: {e}")

# Test with 2-tuple (problematic format that was causing errors)
try:
    text, model, usage = _safe_unpack_chat_result(("Hello world", "gpt-4"))
    print("SUCCESS: 2-tuple unpacking works (this was the bug!)")
    print(f"  Text: {text}, Model: {model}, Usage: {usage}")
except Exception as e:
    print(f"FAILED: 2-tuple unpacking failed: {e}")

# Test 3: Check if we're using the safe wrapper in LLMRouter
try:
    from shared.llm_router import LLMRouter
    import inspect
    
    # Get the source of complete_for_role method
    source = inspect.getsource(LLMRouter.complete_for_role)
    if "safe_chat_complete_sync" in source:
        print("SUCCESS: LLMRouter.complete_for_role uses safe_chat_complete_sync")
    else:
        print("WARNING: LLMRouter may still be using direct chat_complete_sync")
        
    # Check the complete method too
    source_complete = inspect.getsource(LLMRouter.complete)
    if "safe_chat_complete_sync" in source_complete:
        print("SUCCESS: LLMRouter.complete uses safe_chat_complete_sync")
    else:
        print("WARNING: LLMRouter.complete may still be using direct chat_complete_sync")
        
except Exception as e:
    print(f"WARNING: Could not verify LLMRouter methods: {e}")

print("\n=== Fix Verification Complete ===")
print("If all tests show SUCCESS, the tuple unpacking fix is properly deployed!")
