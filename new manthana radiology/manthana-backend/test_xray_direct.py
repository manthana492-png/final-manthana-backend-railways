#!/usr/bin/env python3
"""Direct test of X-ray service to verify the fix."""

import requests
import base64
import json

# Create a simple test image (1x1 pixel PNG)
test_image_data = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

# Test the X-ray service directly
url = "https://manthana492-prod-2--manthana-body-xray-serve.modal.run/analyze/xray"

files = {
    'file': ('test.png', test_image_data, 'image/png'),
    'job_id': 'test-fix-verification',
    'patient_id': 'test-patient',
    'skip_llm_narrative': 'false'  # Make sure narrative is required
}

print("Testing X-ray service with tuple unpacking fix...")
print(f"URL: {url}")

try:
    response = requests.post(url, files=files, timeout=120)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("SUCCESS: X-ray analysis completed!")
        print(f"Detected region: {result.get('detected_region', 'unknown')}")
        print(f"Processing time: {result.get('processing_time_sec', 'unknown')}s")
        
        structures = result.get('structures', {})
        if 'narrative_report' in structures:
            narrative = structures['narrative_report']
            print(f"Narrative generated: {len(narrative)} chars")
            print(f"Preview: {narrative[:200]}...")
            print("SUCCESS: Kimi narrative is working!")
        else:
            print("WARNING: No narrative generated")
            
        models = result.get('models_used', [])
        print(f"Models used: {models}")
        
        if 'OpenRouter-narrative-CXR' in models:
            print("SUCCESS: OpenRouter integration working!")
        else:
            print("WARNING: OpenRouter not in models list")
            
    else:
        print(f"FAILED: Status {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.Timeout:
    print("FAILED: Request timed out")
except requests.exceptions.RequestException as e:
    print(f"FAILED: Request error: {e}")
except Exception as e:
    print(f"FAILED: Unexpected error: {e}")

print("Test completed.")
