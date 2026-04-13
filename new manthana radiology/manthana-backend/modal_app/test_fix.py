"""Test the tuple unpacking fix in Modal environment."""

from __future__ import annotations

import modal
import sys
import os

from modal_app.common import (
    manthana_secret,
    models_volume,
    service_image_body_xray,
)

app = modal.App("manthana-test-fix")

@app.function(
    image=service_image_body_xray(),
    gpu="T4",
    volumes={"/models": models_volume()},
    secrets=[manthana_secret()],
    timeout=300,
)
def test_tuple_fix():
    """Test the tuple unpacking fix in the actual Modal environment."""
    import sys
    import os

    os.environ.setdefault("MANTHANA_LLM_REPO_ROOT", "/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/shared")
    
    print("=== Testing Modal Tuple Unpacking Fix ===")
    
    # Test 1: Import the fixed llm_router
    try:
        from llm_router import llm_router, safe_chat_complete_sync
        print("SUCCESS: llm_router imported with safe_chat_complete_sync")
    except Exception as e:
        print(f"FAILED: Import error: {e}")
        return {"status": "failed", "error": str(e)}
    
    # Test 2: Check manthana-inference version
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
    
    # Test 3: Test OpenRouter completion with vision_primary (the failing role)
    try:
        print("Testing OpenRouter vision_primary completion...")
        result = llm_router.complete_for_role(
            "vision_primary",
            "You are a radiologist analyzing a chest X-ray.",
            "Describe what you see in this image.",
            max_tokens=150,
        )
        print(f"SUCCESS: vision_primary completion worked")
        print(f"Model: {result.get('model_used', 'unknown')}")
        print(f"Content: {result.get('content', 'No content')[:200]}...")
        return {"status": "success", "model": result.get('model_used'), "content_preview": result.get('content', '')[:200]}
        
    except Exception as e:
        print(f"FAILED: vision_primary completion failed: {e}")
        print("Testing fallback to narrative_default...")
        
        try:
            result = llm_router.complete_for_role(
                "narrative_default",
                "You are a radiologist.",
                "Test message to verify the fix is working.",
                max_tokens=100,
            )
            print(f"SUCCESS: narrative_default fallback worked")
            return {"status": "success_fallback", "model": result.get('model_used')}
        except Exception as e2:
            print(f"FAILED: Even fallback failed: {e2}")
            return {"status": "failed", "error": f"vision_primary: {e}, fallback: {e2}"}

if __name__ == "__main__":
    with app.run():
        result = test_tuple_fix.remote()
        print(f"Test result: {result}")
