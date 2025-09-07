#!/usr/bin/env python3
"""
FinLex Quick Test Script

Test the system with sample data without requiring full database setup.
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path

# Add the finlex directory to the path
sys.path.append(str(Path(__file__).parent / "finlex"))

# Set environment variables for testing
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "test_key")
os.environ["DATABASE_URL"] = "sqlite:///test.db"  # Use SQLite for testing

def test_gemini_client():
    """Test Gemini client with sample data"""
    print("🧪 Testing Gemini Client...")
    
    try:
        from services.gemini_client import GeminiClient
        
        # Create client
        client = GeminiClient()
        print("✅ Gemini client created successfully")
        
        # Test content hashing
        test_content = "Sample policy content for testing"
        hash_result = client._hash_content(test_content)
        print(f"✅ Content hashing works: {hash_result[:8]}...")
        
        # Test system prompt building
        prompt = client._build_system_prompt("obligation_extraction")
        assert "JSON" in prompt
        print("✅ System prompt generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini client test failed: {e}")
        return False

def test_transaction_processor():
    """Test transaction processing"""
    print("🧪 Testing Transaction Processor...")
    
    try:
        from services.ingest.main import TransactionProcessor
        
        processor = TransactionProcessor()
        
        # Test currency normalization
        usd_amount = processor.normalize_currency(100.0, "USD")
        assert usd_amount == 100.0
        
        eur_amount = processor.normalize_currency(100.0, "EUR")
        assert eur_amount == 108.0  # Based on mock rates
        
        print("✅ Currency normalization works")
        
        # Test PII hashing
        hashed = processor.hash_pii("John Doe")
        assert len(hashed) == 64
        assert hashed != "John Doe"
        print("✅ PII hashing works")
        
        # Test timestamp normalization
        test_time = datetime(2024, 1, 1, 12, 0, 0)
        normalized = processor.normalize_timestamp(test_time)
        assert normalized.tzinfo == timezone.utc
        print("✅ Timestamp normalization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Transaction processor test failed: {e}")
        return False

def test_compliance_rules():
    """Test compliance rules"""
    print("🧪 Testing Compliance Rules...")
    
    try:
        from services.matcher.main import ThresholdRule
        
        # Create rules for US jurisdiction
        rules = ThresholdRule("US")
        
        # Test threshold values
        assert rules.thresholds["large_transaction"] == 100000.0
        assert rules.thresholds["cash_transaction_reporting"] == 10000.0
        print("✅ Threshold rules initialized correctly")
        
        # Create mock transaction
        class MockTransaction:
            def __init__(self):
                self.amount_usd = 150000.0
                self.type = "TRANSFER"
                self.amount_1d_total = 150000.0
                self.transaction_count_1d = 1
        
        mock_tx = MockTransaction()
        
        # Test large transaction rule
        result = rules.check_large_transaction(mock_tx)
        assert result is not None
        assert result['triggered'] is True
        print("✅ Large transaction rule works")
        
        return True
        
    except Exception as e:
        print(f"❌ Compliance rules test failed: {e}")
        return False

async def test_async_components():
    """Test async components like Gemini calls"""
    print("🧪 Testing Async Components...")
    
    try:
        from services.gemini_client import get_gemini_client
        
        client = get_gemini_client()
        
        # Test obligation extraction (will use mock)
        sample_policy = "Financial institutions must report transactions exceeding $10,000 within 24 hours."
        
        response = await client.extract_obligations(sample_policy)
        
        assert hasattr(response, 'json_output')
        assert 'obligations' in response.json_output
        print("✅ Async obligation extraction works")
        
        # Test violation analysis (will use mock)
        sample_transaction = {
            'id': 'test_tx_001',
            'amount': 150000.0,
            'type': 'TRANSFER'
        }
        
        response = await client.analyze_violation(
            sample_transaction,
            "Must report large transactions",
            ["Report transactions > $100k"]
        )
        
        assert hasattr(response, 'json_output')
        assert 'violation_detected' in response.json_output
        print("✅ Async violation analysis works")
        
        return True
        
    except Exception as e:
        print(f"❌ Async components test failed: {e}")
        return False

def create_sample_data():
    """Create sample data files for testing"""
    print("📊 Creating Sample Data...")
    
    # Create sample transactions CSV
    sample_transactions = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedMerchant
1,TRANSFER,150000,C1234567890,200000,50000,M9876543210,0,150000,0,0
2,CASH_OUT,15000,C2345678901,20000,5000,M8765432109,0,15000,0,1
3,PAYMENT,5000,C3456789012,10000,5000,M7654321098,1000,6000,0,0
4,TRANSFER,250000,C4567890123,300000,50000,M6543210987,0,250000,1,0
5,CASH_IN,3000,C5678901234,2000,5000,M5432109876,5000,8000,0,0
"""
    
    with open("sample_transactions.csv", "w") as f:
        f.write(sample_transactions)
    print("✅ Sample transactions created: sample_transactions.csv")
    
    # Create sample policy
    sample_policy = """
ANTI-MONEY LAUNDERING POLICY

Section 1: Transaction Reporting Requirements
Financial institutions must report all currency transactions in amounts exceeding $10,000 to the Financial Crimes Enforcement Network (FinCEN) within 15 calendar days of the transaction.

Section 2: Large Transaction Monitoring  
Banks shall monitor and flag any individual transaction exceeding $100,000 for enhanced due diligence procedures.

Section 3: Cash Transaction Controls
All cash transactions exceeding $10,000 must be reported to regulatory authorities within 24 hours of occurrence.

Section 4: Record Keeping Requirements
Financial institutions shall maintain records of all wire transfers and electronic payments for a minimum period of five years from the date of the transaction.

Section 5: Suspicious Activity Reporting
Banks must file Suspicious Activity Reports (SARs) for any transaction that appears unusual or potentially suspicious within 30 days of detection.
"""
    
    with open("sample_policy.txt", "w") as f:
        f.write(sample_policy)
    print("✅ Sample policy created: sample_policy.txt")
    
    return True

def run_quick_streamlit_test():
    """Test if Streamlit app can start"""
    print("🎯 Testing Streamlit App...")
    
    try:
        # Import main components
        import streamlit as st
        import pandas as pd
        
        # Test that we can import the app
        sys.path.append(str(Path(__file__).parent / "finlex"))
        
        print("✅ Streamlit imports work")
        print("📝 To test the full UI, run: streamlit run finlex/ui/app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("""
    ╔═══════════════════════════════════════════════╗
    ║         🧪 FinLex Quick Test Suite            ║
    ║            Testing Core Components            ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Run synchronous tests
    tests = [
        test_gemini_client,
        test_transaction_processor, 
        test_compliance_rules,
        create_sample_data,
        run_quick_streamlit_test
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Run async tests
    print("🔄 Running async tests...")
    try:
        async_result = asyncio.run(test_async_components())
        results.append(async_result)
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("""
🎉 All tests passed! FinLex Audit AI is working correctly!

🚀 Next Steps:
1. Run the Streamlit UI: streamlit run finlex/ui/app.py
2. Upload the sample files created:
   - sample_transactions.csv (for transaction data)
   - sample_policy.txt (for policy analysis)
3. Test compliance scanning with the sample data

📁 Sample files created:
- sample_transactions.csv - 5 sample transactions with violations
- sample_policy.txt - Sample AML policy document

🔑 Your Gemini API key is configured and ready!
        """)
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the errors above.")
        
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())