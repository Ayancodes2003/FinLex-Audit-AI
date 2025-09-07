"""
Comprehensive test suite for FinLex Audit AI

Tests for compliance rules, transaction processing, and system integration.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import hashlib
import json

# Import modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.database import Transaction, Obligation, Violation
from services.gemini_client import GeminiClient, GeminiResponse
from services.ingest.main import TransactionProcessor
from services.matcher.main import ThresholdRule, ComplianceMatcher


class TestTransactionProcessor:
    """Test transaction ingestion and normalization"""
    
    @pytest.fixture
    def processor(self):
        return TransactionProcessor()
    
    def test_currency_normalization(self, processor):
        """Test currency conversion to USD"""
        # Test USD (no conversion)
        assert processor.normalize_currency(100.0, "USD") == 100.0
        
        # Test EUR conversion
        eur_amount = processor.normalize_currency(100.0, "EUR")
        assert eur_amount == 108.0  # 100 * 1.08
        
        # Test GBP conversion
        gbp_amount = processor.normalize_currency(100.0, "GBP")
        assert gbp_amount == 125.0  # 100 * 1.25
    
    def test_pii_hashing(self, processor):
        """Test PII data hashing"""
        test_name = "John Doe"
        hashed = processor.hash_pii(test_name)
        
        # Should be deterministic
        assert processor.hash_pii(test_name) == hashed
        
        # Should be different for different inputs
        assert processor.hash_pii("Jane Doe") != hashed
        
        # Should be 64 character hex string
        assert len(hashed) == 64
        assert all(c in '0123456789abcdef' for c in hashed)
    
    def test_timestamp_normalization(self, processor):
        """Test timestamp normalization to UTC"""
        # Test naive datetime (assumes UTC)
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        normalized = processor.normalize_timestamp(naive_dt)
        assert normalized.tzinfo == timezone.utc
        
        # Test timezone-aware datetime
        utc_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        normalized = processor.normalize_timestamp(utc_dt)
        assert normalized.tzinfo == timezone.utc
        assert normalized == utc_dt
    
    def test_transaction_id_generation(self, processor):
        """Test deterministic transaction ID generation"""
        tx_data = {
            'step': 1,
            'nameOrig': 'John Doe',
            'nameDest': 'Jane Doe', 
            'amount': 1000.0,
            'timestamp': '2024-01-01T12:00:00Z'
        }
        
        tx_id1 = processor.generate_transaction_id(tx_data)
        tx_id2 = processor.generate_transaction_id(tx_data)
        
        # Should be deterministic
        assert tx_id1 == tx_id2
        
        # Should change with different data
        tx_data['amount'] = 2000.0
        tx_id3 = processor.generate_transaction_id(tx_data)
        assert tx_id3 != tx_id1


class TestThresholdRules:
    """Test deterministic compliance rules"""
    
    @pytest.fixture
    def us_rules(self):
        return ThresholdRule("US")
    
    @pytest.fixture
    def eu_rules(self):
        return ThresholdRule("EU")
    
    @pytest.fixture
    def sample_transaction(self):
        return Transaction(
            id="test_tx_001",
            step=1,
            type="TRANSFER",
            amount_usd=150000.0,  # Above $100k threshold
            name_orig_hash="hash1",
            name_dest_hash="hash2",
            amount_1d_total=150000.0,
            amount_30d_total=500000.0,
            transaction_count_1d=3,
            transaction_count_30d=25,
            timestamp_utc=datetime.now(timezone.utc)
        )
    
    def test_large_transaction_rule(self, us_rules, sample_transaction):
        """Test large transaction threshold rule"""
        # Should trigger for $150k transaction
        result = us_rules.check_large_transaction(sample_transaction)
        assert result is not None
        assert result['triggered'] is True
        assert result['rule_type'] == 'large_transaction'
        assert result['confidence'] == 1.0
        assert result['risk_level'] in ['medium', 'high']
        
        # Should not trigger for small transaction
        sample_transaction.amount_usd = 50000.0
        result = us_rules.check_large_transaction(sample_transaction)
        assert result is None
    
    def test_cash_reporting_threshold(self, us_rules):
        """Test cash transaction reporting rule"""
        cash_transaction = Transaction(
            id="test_cash_001",
            type="CASH_OUT",
            amount_usd=15000.0,  # Above $10k threshold
            name_orig_hash="hash1",
            name_dest_hash="hash2"
        )
        
        result = us_rules.check_cash_reporting_threshold(cash_transaction)
        assert result is not None
        assert result['rule_type'] == 'cash_reporting'
        assert result['triggered'] is True
        
        # Non-cash transaction should not trigger
        cash_transaction.type = "TRANSFER"
        result = us_rules.check_cash_reporting_threshold(cash_transaction)
        assert result is None
    
    def test_velocity_limits(self, us_rules, sample_transaction):
        """Test transaction velocity limits"""
        # High 1-day velocity should trigger
        sample_transaction.amount_1d_total = 75000.0  # Above $50k threshold
        result = us_rules.check_velocity_limits(sample_transaction)
        assert result is not None
        assert result['rule_type'] == 'velocity_limit'
        
        # Low velocity should not trigger
        sample_transaction.amount_1d_total = 25000.0
        result = us_rules.check_velocity_limits(sample_transaction)
        assert result is None
    
    def test_frequency_limits(self, us_rules, sample_transaction):
        """Test transaction frequency limits"""
        # High frequency should trigger
        sample_transaction.transaction_count_1d = 8  # Above 5 transaction threshold
        result = us_rules.check_frequency_limits(sample_transaction)
        assert result is not None
        assert result['rule_type'] == 'frequency_limit'
        
        # Normal frequency should not trigger
        sample_transaction.transaction_count_1d = 3
        result = us_rules.check_frequency_limits(sample_transaction)
        assert result is None
    
    def test_jurisdiction_differences(self, us_rules, eu_rules):
        """Test different thresholds for different jurisdictions"""
        assert us_rules.thresholds['large_transaction'] == 100000.0
        assert eu_rules.thresholds['large_transaction'] == 85000.0
        
        assert us_rules.thresholds['cash_transaction_reporting'] == 10000.0
        assert eu_rules.thresholds['cash_transaction_reporting'] == 8500.0


class TestGeminiClient:
    """Test Gemini client functionality with mocks"""
    
    @pytest.fixture
    def mock_gemini_client(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            return GeminiClient()
    
    def test_content_hashing(self, mock_gemini_client):
        """Test content hashing for audit logging"""
        content = "Test policy content"
        hash1 = mock_gemini_client._hash_content(content)
        hash2 = mock_gemini_client._hash_content(content)
        
        # Should be deterministic
        assert hash1 == hash2
        
        # Should be 16 character hex string
        assert len(hash1) == 16
        assert all(c in '0123456789abcdef' for c in hash1)
    
    def test_system_prompt_building(self, mock_gemini_client):
        """Test system prompt generation"""
        obligation_prompt = mock_gemini_client._build_system_prompt("obligation_extraction")
        violation_prompt = mock_gemini_client._build_system_prompt("violation_analysis")
        
        # Should contain task-specific content
        assert "obligation" in obligation_prompt.lower()
        assert "violation" in violation_prompt.lower()
        
        # Should have JSON schema requirements
        assert "JSON" in obligation_prompt
        assert "JSON" in violation_prompt
    
    @pytest.mark.asyncio
    async def test_obligation_extraction_mock(self, mock_gemini_client):
        """Test obligation extraction with mock response"""
        policy_text = "Financial institutions must report transactions exceeding $10,000 within 24 hours."
        
        # This will use the mock implementation
        response = await mock_gemini_client.extract_obligations(policy_text)
        
        assert isinstance(response, GeminiResponse)
        assert 'obligations' in response.json_output
        assert len(response.json_output['obligations']) > 0
        assert response.confidence > 0
    
    @pytest.mark.asyncio
    async def test_violation_analysis_mock(self, mock_gemini_client):
        """Test violation analysis with mock response"""
        transaction_data = {
            'id': 'test_tx_001',
            'amount': 150000.0,
            'type': 'TRANSFER'
        }
        
        obligation_context = "Must report large transactions"
        policy_snippets = ["Report transactions > $100k"]
        
        response = await mock_gemini_client.analyze_violation(
            transaction_data, obligation_context, policy_snippets
        )
        
        assert isinstance(response, GeminiResponse)
        assert 'violation_detected' in response.json_output
        assert 'confidence' in response.json_output
        assert 'risk_level' in response.json_output
    
    def test_transaction_sanitization(self, mock_gemini_client):
        """Test PII sanitization before LLM calls"""
        transaction = {
            'id': 'tx_001',
            'nameOrig': 'John Doe',
            'nameDest': 'Jane Smith', 
            'amount': 5000.0,
            'account_id': 'ACC123456',
            'safe_field': 'TRANSFER'
        }
        
        sanitized = mock_gemini_client._sanitize_transaction_data(transaction)
        
        # PII fields should be hashed
        assert sanitized['nameOrig'] != 'John Doe'
        assert sanitized['nameDest'] != 'Jane Smith'
        assert len(sanitized['nameOrig']) == 16  # Hash length
        
        # Safe fields should remain unchanged
        assert sanitized['amount'] == 5000.0
        assert sanitized['safe_field'] == 'TRANSFER'


class TestComplianceMatcher:
    """Test compliance matching integration"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.query.return_value.filter.return_value.all.return_value = []
        return session
    
    @pytest.fixture
    def sample_obligations(self):
        """Sample obligations for testing"""
        return [
            Obligation(
                id="obl_001",
                actor="financial institutions",
                action="report large transactions",
                type="requirement",
                condition="amount > $100,000",
                jurisdiction="US",
                confidence=0.95
            ),
            Obligation(
                id="obl_002", 
                actor="banks",
                action="monitor suspicious activity",
                type="requirement",
                condition="unusual patterns",
                jurisdiction="US",
                confidence=0.90
            )
        ]
    
    @pytest.mark.asyncio
    async def test_full_matching_pipeline(self, mock_db_session, sample_obligations):
        """Test complete matching pipeline"""
        matcher = ComplianceMatcher()
        
        # Mock database responses
        mock_transactions = [
            Transaction(
                id="tx_001",
                amount_usd=150000.0,
                type="TRANSFER",
                amount_1d_total=150000.0,
                transaction_count_1d=1
            )
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.side_effect = [
            mock_transactions,  # Transaction query
            sample_obligations  # Obligations query
        ]
        
        result = await matcher.match_transactions(
            transaction_ids=["tx_001"],
            jurisdiction="US",
            rule_types=["threshold", "semantic"],
            confidence_threshold=0.7,
            db=mock_db_session
        )
        
        assert 'matches' in result
        assert 'summary' in result
        assert 'errors' in result
        
        # Should find threshold violations for large transaction
        matches = result['matches']
        assert len(matches) > 0
        
        # Check for large transaction violation
        large_tx_violations = [m for m in matches if m.get('violation_type') == 'large_transaction']
        assert len(large_tx_violations) > 0


class TestIntegration:
    """Integration tests for end-to-end workflows"""
    
    def test_transaction_processing_workflow(self):
        """Test complete transaction processing workflow"""
        processor = TransactionProcessor()
        
        # Sample transaction input
        raw_transaction = {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 150000.0,
            'currency': 'USD',
            'nameOrig': 'John Doe',
            'nameDest': 'Jane Smith',
            'oldbalanceOrg': 200000.0,
            'newbalanceOrig': 50000.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 150000.0,
            'isFraud': False,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Process transaction
        tx_id = processor.generate_transaction_id({
            'step': raw_transaction['step'],
            'nameOrig': raw_transaction['nameOrig'],
            'nameDest': raw_transaction['nameDest'],
            'amount': raw_transaction['amount'],
            'timestamp': raw_transaction['timestamp'].isoformat()
        })
        
        # Verify processing
        assert tx_id is not None
        assert len(tx_id) == 64  # SHA-256 hash length
        
        # Test currency normalization
        normalized_amount = processor.normalize_currency(
            raw_transaction['amount'], 
            raw_transaction['currency']
        )
        assert normalized_amount == 150000.0  # USD, no conversion
        
        # Test PII hashing
        hashed_orig = processor.hash_pii(raw_transaction['nameOrig'])
        hashed_dest = processor.hash_pii(raw_transaction['nameDest'])
        
        assert hashed_orig != raw_transaction['nameOrig']
        assert hashed_dest != raw_transaction['nameDest']
    
    def test_policy_obligation_extraction_workflow(self):
        """Test policy processing and obligation extraction workflow"""
        # Sample policy text
        policy_text = """
        Section 1: Reporting Requirements
        Financial institutions must report all currency transactions in amounts 
        exceeding $10,000 to the Financial Crimes Enforcement Network (FinCEN) 
        within 15 calendar days of the transaction.
        
        Section 2: Record Keeping
        Banks shall maintain records of all wire transfers for a period of 
        five years from the date of the transaction.
        """
        
        # Test clause splitting (would be part of extractor service)
        from services.extractor.main import PolicyClauseSplitter
        
        splitter = PolicyClauseSplitter()
        clauses = splitter.split_into_clauses(policy_text)
        
        assert len(clauses) > 0
        
        # Should identify clauses with obligation keywords
        obligation_clauses = [c for c in clauses if any(
            keyword in c.lower() for keyword in ['must', 'shall', 'required']
        )]
        assert len(obligation_clauses) >= 2  # At least reporting and record keeping
    
    def test_compliance_scanning_workflow(self):
        """Test end-to-end compliance scanning"""
        # Create sample data
        transaction = Transaction(
            id="integration_test_tx",
            step=1,
            type="TRANSFER",
            amount_usd=125000.0,  # Above threshold
            name_orig_hash="hash_orig",
            name_dest_hash="hash_dest",
            amount_1d_total=125000.0,
            transaction_count_1d=1,
            timestamp_utc=datetime.now(timezone.utc)
        )
        
        # Test threshold rules
        threshold_rules = ThresholdRule("US")
        violations = threshold_rules.check_all_rules(transaction)
        
        # Should detect large transaction violation
        assert len(violations) > 0
        large_tx_violation = next(
            (v for v in violations if v['rule_type'] == 'large_transaction'), 
            None
        )
        assert large_tx_violation is not None
        assert large_tx_violation['triggered'] is True
        assert large_tx_violation['confidence'] == 1.0


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


# Test data fixtures
@pytest.fixture
def sample_paysim_data():
    """Sample Paysim1-like transaction data"""
    return pd.DataFrame([
        {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 150000.0,
            'nameOrig': 'C1234567890',
            'oldbalanceOrg': 200000.0,
            'newbalanceOrig': 50000.0,
            'nameDest': 'M9876543210',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 150000.0,
            'isFraud': 0,
            'isFlaggedMerchant': 0
        },
        {
            'step': 2,
            'type': 'CASH_OUT',
            'amount': 15000.0,
            'nameOrig': 'C2345678901',
            'oldbalanceOrg': 20000.0,
            'newbalanceOrig': 5000.0,
            'nameDest': 'M8765432109',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 15000.0,
            'isFraud': 0,
            'isFlaggedMerchant': 1
        }
    ])


@pytest.fixture
def sample_c3pa_policy():
    """Sample C3PA policy text"""
    return """
    Anti-Money Laundering Policy

    1. Transaction Monitoring
    Financial institutions must monitor all transactions for suspicious activity 
    and report any transactions exceeding $10,000 in cash to regulatory authorities.

    2. Customer Due Diligence
    Banks shall perform enhanced due diligence on all high-risk customers and 
    maintain updated customer information.

    3. Record Retention
    All transaction records must be retained for a minimum period of five years 
    and be readily available for regulatory inspection.
    """


if __name__ == "__main__":
    pytest.main([__file__, "-v"])