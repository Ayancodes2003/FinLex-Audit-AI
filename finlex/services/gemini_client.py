"""
Gemini Client Service for FinLex Audit AI

This module provides a centralized client for all Gemini-Flash LLM interactions,
ensuring consistent formatting, logging, and security compliance.
"""

import json
import hashlib
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass

# TODO: Replace with actual Gemini SDK imports
# from google.generativeai import GenerativeModel
# import google.generativeai as genai

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Structured response from Gemini API"""
    json_output: Dict[str, Any]
    explanation: str
    confidence: float
    model_version: str
    response_time_ms: int


class GeminiClient:
    """
    Production-ready Gemini client with security, logging, and compliance features.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter required")
        
        # TODO: Initialize actual Gemini client
        # genai.configure(api_key=self.api_key)
        # self.model = GenerativeModel('gemini-1.5-flash')
        
        self.model_version = "gemini-1.5-flash"
        
    def _hash_content(self, content: str) -> str:
        """Generate SHA-256 hash for content logging (no PII exposure)"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _log_llm_call(self, prompt_hash: str, response_hash: str, 
                     response_time_ms: int, success: bool):
        """Log LLM call metadata for audit trail"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.model_version,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "response_time_ms": response_time_ms,
            "success": success
        }
        logger.info(f"LLM_CALL: {json.dumps(log_entry)}")
    
    def _build_system_prompt(self, task_type: str) -> str:
        """Build task-specific system prompts"""
        base_prompt = """You are a financial compliance AI assistant. 
You must return responses in this exact format:
1. First, output a valid JSON object containing structured data
2. Then output human-readable explanation text

Always use temperature=0.0 for deterministic compliance outputs.
Ensure all responses are factual and based solely on provided context."""
        
        if task_type == "obligation_extraction":
            return f"""{base_prompt}

Task: Extract structured obligations from policy text.
Output JSON schema:
{{
  "obligations": [
    {{
      "id": "unique_obligation_id",
      "actor": "who must comply",
      "action": "what action is required/prohibited",
      "type": "requirement|prohibition|condition",
      "condition": "when this applies",
      "jurisdiction": "applicable jurisdiction",
      "temporal_scope": "time limits or duration",
      "confidence": 0.0-1.0
    }}
  ]
}}"""
        
        elif task_type == "violation_analysis":
            return f"""{base_prompt}

Task: Analyze potential compliance violations.
Output JSON schema:
{{
  "violation_detected": true/false,
  "matched_obligation_id": "obligation_id",
  "confidence": 0.0-1.0,
  "reasoning": "step-by-step analysis",
  "recommended_action": "immediate|review|monitor|escalate",
  "evidence_ids": ["evidence1", "evidence2"],
  "risk_level": "low|medium|high|critical"
}}"""
        
        return base_prompt
    
    async def extract_obligations(self, policy_text: str, 
                                 few_shot_examples: List[Dict] = None) -> GeminiResponse:
        """
        Extract structured obligations from policy text using few-shot examples from ObliQa
        """
        start_time = datetime.utcnow()
        
        system_prompt = self._build_system_prompt("obligation_extraction")
        
        # Build few-shot context if provided
        examples_context = ""
        if few_shot_examples:
            examples_context = "\nExamples from similar obligations:\n"
            for example in few_shot_examples[:3]:  # Limit to 3 examples
                examples_context += f"Policy: {example.get('policy', 'N/A')}\n"
                examples_context += f"Obligation: {json.dumps(example.get('obligation', {}))}\n\n"
        
        prompt = f"""{system_prompt}

{examples_context}

Policy text to analyze:
{policy_text}

Extract all compliance obligations and return them in the specified JSON format."""
        
        prompt_hash = self._hash_content(prompt)
        
        try:
            # TODO: Replace with actual Gemini API call
            # response = await self.model.generate_content_async(
            #     prompt,
            #     generation_config={"temperature": 0.0, "max_output_tokens": 2048}
            # )
            
            # Mock response for development
            mock_obligations = [
                {
                    "id": f"obl_{self._hash_content(policy_text)[:8]}",
                    "actor": "financial institution",
                    "action": "report transactions exceeding threshold",
                    "type": "requirement",
                    "condition": "transaction amount > $10,000",
                    "jurisdiction": "US",
                    "temporal_scope": "within 24 hours",
                    "confidence": 0.95
                }
            ]
            
            json_output = {"obligations": mock_obligations}
            explanation = "Extracted financial reporting obligation based on policy text analysis."
            
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            response_hash = self._hash_content(json.dumps(json_output))
            
            self._log_llm_call(prompt_hash, response_hash, response_time, True)
            
            return GeminiResponse(
                json_output=json_output,
                explanation=explanation,
                confidence=0.95,
                model_version=self.model_version,
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._log_llm_call(prompt_hash, "error", response_time, False)
            logger.error(f"Gemini obligation extraction failed: {e}")
            raise
    
    async def analyze_violation(self, transaction_data: Dict[str, Any], 
                               obligation_context: str,
                               policy_snippets: List[str]) -> GeminiResponse:
        """
        Analyze potential compliance violation using RAG context
        """
        start_time = datetime.utcnow()
        
        system_prompt = self._build_system_prompt("violation_analysis")
        
        # Sanitize transaction data (hash PII fields)
        sanitized_transaction = self._sanitize_transaction_data(transaction_data)
        
        context = f"""
Policy context:
{obligation_context}

Relevant policy snippets:
{' | '.join(policy_snippets)}

Transaction data (PII hashed):
{json.dumps(sanitized_transaction, indent=2)}
"""
        
        prompt = f"""{system_prompt}

{context}

Analyze whether this transaction violates any compliance obligations and return your analysis in the specified JSON format."""
        
        prompt_hash = self._hash_content(prompt)
        
        try:
            # TODO: Replace with actual Gemini API call
            # response = await self.model.generate_content_async(
            #     prompt,
            #     generation_config={"temperature": 0.0, "max_output_tokens": 1024}
            # )
            
            # Mock response for development
            amount = sanitized_transaction.get('amount', 0)
            violation_detected = amount > 100000  # Simple threshold rule
            
            json_output = {
                "violation_detected": violation_detected,
                "matched_obligation_id": f"obl_{self._hash_content(obligation_context)[:8]}",
                "confidence": 0.87 if violation_detected else 0.23,
                "reasoning": f"Transaction amount ${amount:,.2f} {'exceeds' if violation_detected else 'is below'} regulatory threshold",
                "recommended_action": "escalate" if violation_detected else "monitor",
                "evidence_ids": ["policy_snippet_1", "transaction_data"],
                "risk_level": "high" if violation_detected else "low"
            }
            
            explanation = f"""
Analysis of transaction shows {'a potential violation' if violation_detected else 'no violation'}.
The transaction amount of ${amount:,.2f} {'exceeds' if violation_detected else 'falls within'} 
the regulatory threshold requiring additional reporting.
Recommended action: {json_output['recommended_action']}.
"""
            
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            response_hash = self._hash_content(json.dumps(json_output))
            
            self._log_llm_call(prompt_hash, response_hash, response_time, True)
            
            return GeminiResponse(
                json_output=json_output,
                explanation=explanation.strip(),
                confidence=json_output["confidence"],
                model_version=self.model_version,
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._log_llm_call(prompt_hash, "error", response_time, False)
            logger.error(f"Gemini violation analysis failed: {e}")
            raise
    
    def _sanitize_transaction_data(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hash PII fields in transaction data before sending to LLM
        """
        sanitized = transaction.copy()
        
        # Hash PII fields
        pii_fields = ['nameOrig', 'nameDest', 'account_id', 'customer_id']
        for field in pii_fields:
            if field in sanitized:
                sanitized[field] = self._hash_content(str(sanitized[field]))
        
        return sanitized


# Singleton instance for application use
gemini_client = None

def get_gemini_client() -> GeminiClient:
    """Get singleton Gemini client instance"""
    global gemini_client
    if gemini_client is None:
        gemini_client = GeminiClient()
    return gemini_client