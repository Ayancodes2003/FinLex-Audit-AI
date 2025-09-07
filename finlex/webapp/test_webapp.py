#!/usr/bin/env python3
"""
Test script for FinLex Audit AI Web Application

This script tests the main API endpoints and functionality.
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, method="GET", data=None, description=""):
    """Test a specific endpoint"""
    print(f"\n🧪 Testing: {description}")
    print(f"📡 {method} {BASE_URL}{endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=data)
        
        print(f"✅ Status: {response.status_code}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            result = response.json()
            print(f"📄 Response: {json.dumps(result, indent=2)[:200]}...")
        else:
            print(f"📄 Response: {response.text[:100]}...")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🏦 FinLex Audit AI - Web Application Test Suite")
    print("=" * 60)
    
    # Test basic endpoints
    tests = [
        ("/api/health", "GET", None, "Health Check"),
        ("/api/dashboard/stats", "GET", None, "Dashboard Statistics"),
        ("/api/demo/generate-sample-data", "POST", {}, "Generate Sample Data"),
        ("/api/dashboard/stats", "GET", None, "Dashboard Statistics (after sample data)"),
    ]
    
    results = []
    for endpoint, method, data, description in tests:
        success = test_endpoint(endpoint, method, data, description)
        results.append((description, success))
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {desc}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your FinLex web application is ready!")
        print("\n🌐 You can now:")
        print("  • Access the web interface at http://localhost:5000")
        print("  • Upload transaction files (CSV format)")
        print("  • Upload policy documents (TXT/MD)")
        print("  • Run compliance scans")
        print("  • Review violations and generate reports")
    else:
        print("⚠️  Some tests failed. Please check the application logs.")

if __name__ == "__main__":
    main()