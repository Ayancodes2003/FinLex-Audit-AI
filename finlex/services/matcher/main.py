"""
Compliance Matcher Service

FastAPI service for detecting potential compliance violations using
deterministic rules and semantic matching with FAISS vector search.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
import hashlib
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_session_maker, Transaction, Obligation, Violation, AuditLog
from gemini_client import get_gemini_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinLex Compliance Matcher Service",
    description="Detect compliance violations using rule engine and semantic matching",
    version="1.0.0"
)

# Database dependency
SessionLocal = get_session_maker()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class MatchRequest(BaseModel):
    """Request for compliance matching"""
    transaction_ids: List[str] = Field(..., description="Transaction IDs to analyze")
    jurisdiction: Optional[str] = Field(default="US", description="Jurisdiction to check")
    rule_types: List[str] = Field(default=["threshold", "semantic"], description="Types of rules to apply")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for violations")


class ViolationMatch(BaseModel):
    """Detected violation match"""
    transaction_id: str
    obligation_id: str
    violation_type: str
    confidence: float
    risk_level: str
    reasoning: str
    evidence_ids: List[str]
    recommended_action: str


class MatchResponse(BaseModel):
    """Response for compliance matching"""
    matches: List[ViolationMatch]
    summary: Dict[str, Any]
    success: bool
    errors: List[str]


class ThresholdRule:
    """Deterministic threshold-based compliance rules"""
    
    # Default thresholds (configurable by jurisdiction/company)
    DEFAULT_THRESHOLDS = {
        "US": {
            "large_transaction": 100000.0,  # $100k USD
            "cash_transaction_reporting": 10000.0,  # $10k USD
            "suspicious_frequency": 5,  # 5+ transactions per day
            "high_velocity": 50000.0,  # $50k in 24h window
            "cross_border": 3000.0,  # $3k for international
        },
        "EU": {
            "large_transaction": 85000.0,  # ~€75k
            "cash_transaction_reporting": 8500.0,  # ~€7.5k
            "suspicious_frequency": 5,
            "high_velocity": 42500.0,
            "cross_border": 2500.0,
        }
    }
    
    def __init__(self, jurisdiction: str = "US"):
        self.jurisdiction = jurisdiction
        self.thresholds = self.DEFAULT_THRESHOLDS.get(jurisdiction, self.DEFAULT_THRESHOLDS["US"])
    
    def check_large_transaction(self, transaction: Transaction) -> Optional[Dict[str, Any]]:
        """Check if transaction exceeds large transaction threshold"""
        
        threshold = self.thresholds["large_transaction"]
        if transaction.amount_usd > threshold:
            return {
                "rule_type": "large_transaction",
                "triggered": True,
                "threshold": threshold,
                "actual_value": transaction.amount_usd,
                "confidence": 1.0,
                "reasoning": f"Transaction amount ${transaction.amount_usd:,.2f} exceeds large transaction threshold of ${threshold:,.2f}",
                "risk_level": "high" if transaction.amount_usd > threshold * 2 else "medium"
            }
        return None
    
    def check_cash_reporting_threshold(self, transaction: Transaction) -> Optional[Dict[str, Any]]:
        """Check if cash transaction requires reporting"""
        
        if transaction.type in ['CASH_OUT', 'CASH_IN']:
            threshold = self.thresholds["cash_transaction_reporting"]
            if transaction.amount_usd > threshold:
                return {
                    "rule_type": "cash_reporting",
                    "triggered": True,
                    "threshold": threshold,
                    "actual_value": transaction.amount_usd,
                    "confidence": 1.0,
                    "reasoning": f"Cash transaction of ${transaction.amount_usd:,.2f} exceeds reporting threshold of ${threshold:,.2f}",
                    "risk_level": "medium"
                }
        return None
    
    def check_velocity_limits(self, transaction: Transaction) -> Optional[Dict[str, Any]]:
        """Check if account velocity exceeds limits"""
        
        # Check 1-day velocity
        threshold = self.thresholds["high_velocity"]
        if transaction.amount_1d_total > threshold:
            return {
                "rule_type": "velocity_limit",
                "triggered": True,
                "threshold": threshold,
                "actual_value": transaction.amount_1d_total,
                "confidence": 0.9,
                "reasoning": f"24-hour transaction volume of ${transaction.amount_1d_total:,.2f} exceeds velocity limit of ${threshold:,.2f}",
                "risk_level": "high"
            }
        return None
    
    def check_frequency_limits(self, transaction: Transaction) -> Optional[Dict[str, Any]]:
        """Check if transaction frequency is suspicious"""
        
        threshold = self.thresholds["suspicious_frequency"]
        if transaction.transaction_count_1d > threshold:
            return {
                "rule_type": "frequency_limit",
                "triggered": True,
                "threshold": threshold,
                "actual_value": transaction.transaction_count_1d,
                "confidence": 0.8,
                "reasoning": f"Daily transaction count of {transaction.transaction_count_1d} exceeds suspicious frequency threshold of {threshold}",
                "risk_level": "medium"
            }
        return None
    
    def check_all_rules(self, transaction: Transaction) -> List[Dict[str, Any]]:
        """Apply all threshold rules to a transaction"""
        
        violations = []
        
        checks = [
            self.check_large_transaction,
            self.check_cash_reporting_threshold,
            self.check_velocity_limits,
            self.check_frequency_limits
        ]
        
        for check in checks:
            result = check(transaction)
            if result:
                violations.append(result)
        
        return violations


class VectorMatcher:
    """Semantic matching using FAISS vector search"""
    
    def __init__(self):
        # TODO: Initialize FAISS index with actual embeddings
        # In production, load pre-built FAISS index with obligation embeddings
        self.index = None
        self.obligation_embeddings = {}
        self.mock_embeddings = True
    
    def _get_transaction_embedding(self, transaction: Transaction) -> np.ndarray:
        """Generate embedding for transaction (mock implementation)"""
        
        # Mock embedding based on transaction features
        # In production, use proper embedding model
        features = [
            transaction.amount_usd / 100000.0,  # Normalized amount
            1.0 if transaction.type == 'TRANSFER' else 0.0,
            1.0 if transaction.type == 'PAYMENT' else 0.0,
            1.0 if transaction.type == 'CASH_OUT' else 0.0,
            transaction.amount_1d_total / 100000.0,
            transaction.transaction_count_1d / 10.0,
            1.0 if transaction.is_flagged_merchant else 0.0
        ]
        
        # Pad or truncate to fixed dimension
        embedding = np.array(features + [0.0] * (128 - len(features)))[:128]
        return embedding
    
    def _get_obligation_embedding(self, obligation: Obligation) -> np.ndarray:
        """Generate embedding for obligation (mock implementation)"""
        
        # Mock embedding based on obligation features
        # In production, use actual text embeddings of obligation.action and obligation.condition
        obligation_text = f"{obligation.actor} {obligation.action} {obligation.condition or ''}"
        
        # Simple hash-based mock embedding
        hash_value = hashlib.sha256(obligation_text.encode()).hexdigest()
        hash_int = int(hash_value[:16], 16)
        
        # Convert to float array
        embedding = np.array([
            (hash_int >> (i * 4)) & 0xF for i in range(32)
        ], dtype=np.float32)
        
        # Normalize and extend to 128 dimensions
        embedding = embedding / np.linalg.norm(embedding)
        embedding = np.pad(embedding, (0, 96), mode='constant')
        
        return embedding
    
    def find_similar_obligations(self, transaction: Transaction, 
                               obligations: List[Obligation],
                               top_k: int = 5) -> List[Tuple[Obligation, float]]:
        """Find obligations most similar to transaction"""
        
        if not obligations:
            return []
        
        tx_embedding = self._get_transaction_embedding(transaction)
        
        similarities = []
        for obligation in obligations:
            obl_embedding = self._get_obligation_embedding(obligation)
            
            # Compute cosine similarity
            similarity = np.dot(tx_embedding, obl_embedding) / (
                np.linalg.norm(tx_embedding) * np.linalg.norm(obl_embedding)
            )
            
            similarities.append((obligation, float(similarity)))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def semantic_match(self, transaction: Transaction, 
                      obligations: List[Obligation],
                      confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic matching between transaction and obligations"""
        
        similar_obligations = self.find_similar_obligations(transaction, obligations)
        
        matches = []
        for obligation, similarity in similar_obligations:
            if similarity >= confidence_threshold:
                matches.append({
                    "rule_type": "semantic_match",
                    "obligation_id": obligation.id,
                    "similarity": similarity,
                    "confidence": similarity,
                    "reasoning": f"Transaction semantically similar to obligation '{obligation.action}' (similarity: {similarity:.3f})",
                    "risk_level": "high" if similarity > 0.9 else "medium" if similarity > 0.8 else "low",
                    "obligation_details": {
                        "actor": obligation.actor,
                        "action": obligation.action,
                        "type": obligation.type,
                        "condition": obligation.condition
                    }
                })
        
        return matches


class ComplianceMatcher:
    """Main compliance matching orchestrator"""
    
    def __init__(self):
        self.threshold_matcher = ThresholdRule()
        self.vector_matcher = VectorMatcher()
        self.gemini_client = get_gemini_client()
    
    async def match_transactions(self, transaction_ids: List[str], 
                               jurisdiction: str = "US",
                               rule_types: List[str] = ["threshold", "semantic"],
                               confidence_threshold: float = 0.7,
                               db: Session = None) -> Dict[str, Any]:
        """Match transactions against compliance rules"""
        
        logger.info(f"Matching {len(transaction_ids)} transactions against {rule_types} rules")
        
        # Update threshold matcher for jurisdiction
        self.threshold_matcher = ThresholdRule(jurisdiction)
        
        # Fetch transactions
        transactions = db.query(Transaction).filter(
            Transaction.id.in_(transaction_ids)
        ).all()
        
        if not transactions:
            return {"matches": [], "summary": {"error": "No transactions found"}, "errors": ["No transactions found"]}
        
        # Fetch relevant obligations for semantic matching
        obligations = []
        if "semantic" in rule_types:
            obligations = db.query(Obligation).filter(
                or_(
                    Obligation.jurisdiction == jurisdiction,
                    Obligation.jurisdiction.is_(None)
                )
            ).all()
        
        all_matches = []
        errors = []
        
        for transaction in transactions:
            try:
                tx_matches = await self._match_single_transaction(
                    transaction, obligations, rule_types, confidence_threshold
                )
                all_matches.extend(tx_matches)
                
            except Exception as e:
                error_msg = f"Failed to match transaction {transaction.id}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Generate summary
        summary = self._generate_summary(all_matches, len(transactions))
        
        return {
            "matches": all_matches,
            "summary": summary,
            "errors": errors
        }
    
    async def _match_single_transaction(self, transaction: Transaction,
                                      obligations: List[Obligation],
                                      rule_types: List[str],
                                      confidence_threshold: float) -> List[Dict[str, Any]]:
        """Match a single transaction against all rules"""
        
        matches = []
        
        # Apply threshold rules
        if "threshold" in rule_types:
            threshold_violations = self.threshold_matcher.check_all_rules(transaction)
            for violation in threshold_violations:
                match = {
                    "transaction_id": transaction.id,
                    "obligation_id": f"threshold_{violation['rule_type']}",
                    "violation_type": violation["rule_type"],
                    "confidence": violation["confidence"],
                    "risk_level": violation["risk_level"],
                    "reasoning": violation["reasoning"],
                    "evidence_ids": ["transaction_data", "threshold_rule"],
                    "recommended_action": "review" if violation["risk_level"] == "medium" else "escalate"
                }
                matches.append(match)
        
        # Apply semantic matching
        if "semantic" in rule_types and obligations:
            semantic_matches = self.vector_matcher.semantic_match(
                transaction, obligations, confidence_threshold
            )
            
            for semantic_match in semantic_matches:
                match = {
                    "transaction_id": transaction.id,
                    "obligation_id": semantic_match["obligation_id"],
                    "violation_type": "semantic_violation",
                    "confidence": semantic_match["confidence"],
                    "risk_level": semantic_match["risk_level"],
                    "reasoning": semantic_match["reasoning"],
                    "evidence_ids": ["transaction_data", "semantic_similarity"],
                    "recommended_action": "review" if semantic_match["confidence"] < 0.9 else "escalate"
                }
                matches.append(match)
        
        return matches
    
    def _generate_summary(self, matches: List[Dict], transaction_count: int) -> Dict[str, Any]:
        """Generate summary statistics for matching results"""
        
        if not matches:
            return {
                "total_transactions": transaction_count,
                "total_violations": 0,
                "violation_types": {},
                "risk_levels": {},
                "avg_confidence": 0.0
            }
        
        # Count by violation type
        violation_types = {}
        risk_levels = {}
        confidences = []
        
        for match in matches:
            vtype = match["violation_type"]
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
            
            risk = match["risk_level"]
            risk_levels[risk] = risk_levels.get(risk, 0) + 1
            
            confidences.append(match["confidence"])
        
        return {
            "total_transactions": transaction_count,
            "total_violations": len(matches),
            "violation_types": violation_types,
            "risk_levels": risk_levels,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "high_risk_count": risk_levels.get("high", 0),
            "critical_risk_count": risk_levels.get("critical", 0)
        }


matcher = ComplianceMatcher()


@app.post("/match", response_model=MatchResponse)
async def match_compliance(
    request: MatchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Match transactions against compliance rules"""
    
    logger.info(f"Processing match request for {len(request.transaction_ids)} transactions")
    
    try:
        result = await matcher.match_transactions(
            transaction_ids=request.transaction_ids,
            jurisdiction=request.jurisdiction,
            rule_types=request.rule_types,
            confidence_threshold=request.confidence_threshold,
            db=db
        )
        
        # Convert to response format
        violation_matches = [
            ViolationMatch(**match) for match in result["matches"]
        ]
        
        # Log audit event
        background_tasks.add_task(
            log_matching_audit,
            transaction_count=len(request.transaction_ids),
            violation_count=len(violation_matches),
            summary=result["summary"]
        )
        
        return MatchResponse(
            matches=violation_matches,
            summary=result["summary"],
            success=True,
            errors=result.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"Matching failed: {e}")
        raise HTTPException(status_code=500, detail=f"Matching error: {str(e)}")


@app.post("/match/single/{transaction_id}")
async def match_single_transaction(
    transaction_id: str,
    jurisdiction: str = "US",
    rule_types: List[str] = ["threshold", "semantic"],
    confidence_threshold: float = 0.7,
    db: Session = Depends(get_db)
):
    """Match a single transaction against compliance rules"""
    
    request = MatchRequest(
        transaction_ids=[transaction_id],
        jurisdiction=jurisdiction,
        rule_types=rule_types,
        confidence_threshold=confidence_threshold
    )
    
    return await match_compliance(request, BackgroundTasks(), db)


@app.get("/rules/thresholds/{jurisdiction}")
async def get_thresholds(jurisdiction: str = "US"):
    """Get threshold rules for jurisdiction"""
    
    threshold_rules = ThresholdRule(jurisdiction)
    return {
        "jurisdiction": jurisdiction,
        "thresholds": threshold_rules.thresholds,
        "available_jurisdictions": list(ThresholdRule.DEFAULT_THRESHOLDS.keys())
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "matcher", "timestamp": datetime.utcnow().isoformat()}


async def log_matching_audit(transaction_count: int, violation_count: int, summary: Dict):
    """Log matching audit event"""
    try:
        db = SessionLocal()
        
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            event_type="compliance_matching",
            entity_type="transaction",
            event_data={
                "transaction_count": transaction_count,
                "violation_count": violation_count,
                "summary": summary
            }
        )
        
        db.add(audit_log)
        db.commit()
        db.close()
        
    except Exception as e:
        logger.error(f"Failed to log matching audit: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)