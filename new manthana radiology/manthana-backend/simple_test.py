import urllib.request
import json

print("Testing Modal service connectivity...")

try:
    # Test health endpoint
    url = "https://manthana492-prod-2--manthana-body-xray-serve.modal.run/health"
    print(f"Testing: {url}")
    
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read().decode('utf-8')
        print(f"Health check response: {data}")
        
    print("SUCCESS: Service is reachable!")
    
except Exception as e:
    print(f"FAILED: {e}")
