# Simple test to verify fix
with open("test_output.txt", "w") as f:
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
        from shared.llm_router import _safe_unpack_chat_result
        
        # Test the problematic 2-tuple case
        text, model, usage = _safe_unpack_chat_result(("Hello", "gpt-4"))
        f.write("SUCCESS: 2-tuple unpacking works!\n")
        f.write(f"Text: {text}, Model: {model}, Usage: {usage}\n")
        f.write("The tuple unpacking bug is FIXED!\n")
        
    except Exception as e:
        f.write(f"FAILED: {e}\n")
