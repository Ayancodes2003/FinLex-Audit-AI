"""
RAG Generator Service

FastAPI service for generating explainable compliance violation reports
using Retrieval-Augmented Generation with Gemini-Flash.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import hashlib
import uuid
from sqlalchemy.orm import Session

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_session_maker, Transaction, Obligation, Violation, PolicyDocument, AuditLog
from gemini_client import get_gemini_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinLex RAG Generator Service",
    description="Generate explainable compliance violation reports using RAG",
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


class ExplanationRequest(BaseModel):
    """Request for violation explanation generation"""
    transaction_id: str = Field(..., description="Transaction ID to analyze")
    obligation_id: str = Field(..., description="Relevant obligation ID")
    violation_type: str = Field(..., description="Type of violation detected")
    confidence: float = Field(..., description="Initial confidence score")
    additional_context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")


class ViolationReport(BaseModel):
    """Generated violation report"""
    violation_id: str
    violation_detected: bool
    matched_obligation_id: str
    confidence: float
    reasoning: str
    recommended_action: str
    evidence_ids: List[str]
    risk_level: str
    human_explanation: str
    regulatory_references: List[str]
    next_steps: List[str]


class ExplanationResponse(BaseModel):
    """Response for explanation generation"""
    report: ViolationReport
    evidence_details: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool
    errors: List[str]


class EvidenceCollector:
    """Collect and organize evidence for violation analysis"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def collect_transaction_evidence(self, transaction_id: str) -> Dict[str, Any]:
        """Collect transaction-related evidence"""
        
        transaction = self.db.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if not transaction:
            return {}
        
        # Sanitize transaction data (hash PII)
        evidence = {
            "transaction": {
                "id": transaction.id,
                "type": transaction.type,
                "amount_usd": transaction.amount_usd,
                "timestamp_utc": transaction.timestamp_utc.isoformat(),
                "is_fraud": transaction.is_fraud,
                "is_flagged_merchant": transaction.is_flagged_merchant,
                "derived_features": {
                    "amount_1d_total": transaction.amount_1d_total,
                    "amount_30d_total": transaction.amount_30d_total,
                    "transaction_count_1d": transaction.transaction_count_1d,
                    "transaction_count_30d": transaction.transaction_count_30d
                }
            },
            "account_patterns": {
                "daily_volume": transaction.amount_1d_total,
                "monthly_volume": transaction.amount_30d_total,
                "transaction_frequency": transaction.transaction_count_1d,
                "merchant_flag": transaction.is_flagged_merchant
            }
        }
        
        return evidence
    
    def collect_obligation_evidence(self, obligation_id: str) -> Dict[str, Any]:
        """Collect obligation and policy evidence"""
        
        obligation = self.db.query(Obligation).filter(
            Obligation.id == obligation_id
        ).first()
        
        if not obligation:
            return {}
        
        # Get related policy document
        policy_doc = self.db.query(PolicyDocument).filter(
            PolicyDocument.id == obligation.policy_document_id
        ).first()
        
        evidence = {
            "obligation": {
                "id": obligation.id,
                "actor": obligation.actor,
                "action": obligation.action,
                "type": obligation.type,
                "condition": obligation.condition,
                "jurisdiction": obligation.jurisdiction,
                "temporal_scope": obligation.temporal_scope,
                "confidence": obligation.confidence,
                "source_clause": obligation.source_clause
            },
            "policy_context": {
                "policy_id": policy_doc.id if policy_doc else None,
                "policy_title": policy_doc.title if policy_doc else "Unknown",
                "jurisdiction": policy_doc.jurisdiction if policy_doc else "Unknown",
                "effective_date": policy_doc.effective_date.isoformat() if policy_doc and policy_doc.effective_date else None
            }
        }
        
        return evidence
    
    def collect_similar_cases(self, transaction_id: str, obligation_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Collect similar violation cases for context"""
        
        # Find similar violations based on obligation and transaction type
        transaction = self.db.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if not transaction:
            return []
        
        similar_violations = self.db.query(Violation).join(Transaction).filter(
            and_(
                Violation.obligation_id == obligation_id,
                Transaction.type == transaction.type,
                Violation.id != transaction_id  # Exclude current transaction
            )
        ).limit(limit).all()
        
        cases = []
        for violation in similar_violations:
            cases.append({
                "violation_id": violation.id,
                "confidence": violation.confidence,
                "risk_level": violation.risk_level,
                "reasoning": violation.reasoning[:200] + "..." if len(violation.reasoning) > 200 else violation.reasoning,
                "review_status": violation.review_status
            })
        
        return cases
    
    def collect_regulatory_context(self, obligation: Obligation) -> List[str]:
        """Collect relevant regulatory references"""
        
        # Mock regulatory references based on obligation type and jurisdiction
        regulatory_refs = []
        
        if obligation.jurisdiction == "US":
            if "report" in obligation.action.lower():
                regulatory_refs.append("31 CFR 1010.311 - Currency Transaction Reports")
                regulatory_refs.append("Bank Secrecy Act (BSA) Reporting Requirements")
            
            if "transaction" in obligation.action.lower() and "10000" in str(obligation.condition or ""):
                regulatory_refs.append("31 CFR 1010.311 - $10,000 Currency Transaction Threshold")
            
            if "suspicious" in obligation.action.lower():
                regulatory_refs.append("31 CFR 1023.320 - Suspicious Activity Reports")
                
        elif obligation.jurisdiction == "EU":
            regulatory_refs.append("EU Directive 2015/849 - 4th Anti-Money Laundering Directive")
            regulatory_refs.append("EU Regulation 2015/847 - Wire Transfer Regulation")
        
        # Add general references
        if obligation.type == "requirement":
            regulatory_refs.append("Financial Action Task Force (FATF) Recommendations")
        
        return regulatory_refs


class ViolationAnalyzer:
    """Generate detailed violation analysis using RAG"""
    
    def __init__(self):
        self.gemini_client = get_gemini_client()
    
    async def generate_violation_report(self, transaction_id: str, obligation_id: str,
                                      violation_type: str, initial_confidence: float,
                                      evidence: Dict[str, Any]) -> ViolationReport:
        """Generate comprehensive violation report using Gemini RAG"""
        
        logger.info(f"Generating violation report for transaction {transaction_id}")
        
        # Prepare context for Gemini
        obligation_context = evidence.get("obligation", {})
        transaction_context = evidence.get("transaction", {})
        policy_snippets = self._extract_policy_snippets(evidence)
        
        # Call Gemini for analysis
        gemini_response = await self.gemini_client.analyze_violation(
            transaction_data=transaction_context,
            obligation_context=str(obligation_context),
            policy_snippets=policy_snippets
        )
        
        # Generate violation ID
        violation_id = self._generate_violation_id(transaction_id, obligation_id)
        
        # Extract structured response
        analysis = gemini_response.json_output
        
        # Generate human explanation
        human_explanation = self._generate_human_explanation(
            evidence, analysis, gemini_response.explanation
        )
        
        # Determine regulatory references
        regulatory_refs = evidence.get("regulatory_context", [])
        
        # Generate next steps
        next_steps = self._generate_next_steps(analysis, evidence)
        
        return ViolationReport(
            violation_id=violation_id,
            violation_detected=analysis.get("violation_detected", False),
            matched_obligation_id=obligation_id,
            confidence=analysis.get("confidence", initial_confidence),
            reasoning=analysis.get("reasoning", "Analysis unavailable"),
            recommended_action=analysis.get("recommended_action", "review"),
            evidence_ids=analysis.get("evidence_ids", ["transaction_data", "policy_context"]),
            risk_level=analysis.get("risk_level", "medium"),
            human_explanation=human_explanation,
            regulatory_references=regulatory_refs,
            next_steps=next_steps
        )
    
    def _extract_policy_snippets(self, evidence: Dict[str, Any]) -> List[str]:
        """Extract relevant policy text snippets"""
        
        snippets = []
        
        # Add obligation source clause
        obligation = evidence.get("obligation", {})
        if obligation.get("source_clause"):
            snippets.append(obligation["source_clause"])
        
        # Add relevant policy context
        policy_context = evidence.get("policy_context", {})
        if policy_context.get("policy_title"):
            snippets.append(f"Policy: {policy_context['policy_title']}")
        
        return snippets
    
    def _generate_violation_id(self, transaction_id: str, obligation_id: str) -> str:
        """Generate unique violation ID"""
        content = f"{transaction_id}_{obligation_id}_{datetime.utcnow().isoformat()}"
        return f"viol_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    def _generate_human_explanation(self, evidence: Dict[str, Any], 
                                   analysis: Dict[str, Any], 
                                   llm_explanation: str) -> str:
        """Generate human-readable explanation"""
        
        transaction = evidence.get("transaction", {})
        obligation = evidence.get("obligation", {})
        
        explanation = f"""
COMPLIANCE VIOLATION ANALYSIS

Transaction Details:
- Transaction ID: {transaction.get('id', 'Unknown')}
- Type: {transaction.get('type', 'Unknown')}
- Amount: ${transaction.get('amount_usd', 0):,.2f} USD
- Date: {transaction.get('timestamp_utc', 'Unknown')}

Applicable Regulation:
- Obligation: {obligation.get('action', 'Unknown requirement')}
- Actor: {obligation.get('actor', 'Unknown')}
- Jurisdiction: {obligation.get('jurisdiction', 'Unknown')}

Violation Assessment:
{llm_explanation}

Risk Level: {analysis.get('risk_level', 'Medium').upper()}
Confidence: {analysis.get('confidence', 0.5):.1%}

Recommended Action: {analysis.get('recommended_action', 'Review').upper()}
"""
        
        return explanation.strip()
    
    def _generate_next_steps(self, analysis: Dict[str, Any], evidence: Dict[str, Any]) -> List[str]:
        """Generate actionable next steps"""
        
        steps = []
        
        recommended_action = analysis.get("recommended_action", "review")
        risk_level = analysis.get("risk_level", "medium")
        
        if recommended_action == "escalate" or risk_level == "high":
            steps.extend([
                "Immediately escalate to compliance officer",
                "Initiate enhanced due diligence procedures",
                "Consider filing Suspicious Activity Report (SAR)"
            ])
        elif recommended_action == "review":
            steps.extend([
                "Conduct detailed manual review",
                "Gather additional transaction context",
                "Document review findings"
            ])
        elif recommended_action == "monitor":
            steps.extend([
                "Add account to enhanced monitoring list",
                "Review similar transactions from same account",
                "Set up automated alerts for future activity"
            ])
        
        # Add regulatory-specific steps
        obligation = evidence.get("obligation", {})
        if "report" in obligation.get("action", "").lower():
            steps.append("Prepare regulatory filing if violation confirmed")
        
        if evidence.get("transaction", {}).get("is_fraud"):
            steps.append("Coordinate with fraud prevention team")
        
        return steps


analyzer = ViolationAnalyzer()


@app.post("/generate", response_model=ExplanationResponse)
async def generate_explanation(
    request: ExplanationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Generate explainable violation report"""
    
    logger.info(f"Generating explanation for transaction {request.transaction_id}")
    
    try:
        # Collect evidence
        evidence_collector = EvidenceCollector(db)
        
        transaction_evidence = evidence_collector.collect_transaction_evidence(request.transaction_id)
        obligation_evidence = evidence_collector.collect_obligation_evidence(request.obligation_id)
        similar_cases = evidence_collector.collect_similar_cases(
            request.transaction_id, request.obligation_id
        )
        
        # Get obligation for regulatory context
        obligation = db.query(Obligation).filter(
            Obligation.id == request.obligation_id
        ).first()
        
        regulatory_context = []
        if obligation:
            regulatory_context = evidence_collector.collect_regulatory_context(obligation)
        
        # Combine all evidence
        combined_evidence = {
            **transaction_evidence,
            **obligation_evidence,
            "similar_cases": similar_cases,
            "regulatory_context": regulatory_context,
            "additional_context": request.additional_context
        }
        
        if not transaction_evidence or not obligation_evidence:
            raise HTTPException(status_code=404, detail="Transaction or obligation not found")
        
        # Generate violation report
        violation_report = await analyzer.generate_violation_report(
            transaction_id=request.transaction_id,
            obligation_id=request.obligation_id,
            violation_type=request.violation_type,
            initial_confidence=request.confidence,
            evidence=combined_evidence
        )
        
        # Save violation to database if detected
        if violation_report.violation_detected:
            violation_record = Violation(
                id=violation_report.violation_id,
                transaction_id=request.transaction_id,
                obligation_id=request.obligation_id,
                violation_type=request.violation_type,
                confidence=violation_report.confidence,
                risk_level=violation_report.risk_level,
                reasoning=violation_report.reasoning,
                recommended_action=violation_report.recommended_action,
                evidence_ids=violation_report.evidence_ids,
                llm_explanation=violation_report.human_explanation,
                model_version="gemini-1.5-flash"
            )
            
            db.add(violation_record)
            db.commit()
        
        # Log audit event
        background_tasks.add_task(
            log_explanation_audit,
            violation_id=violation_report.violation_id,
            transaction_id=request.transaction_id,
            obligation_id=request.obligation_id,
            violation_detected=violation_report.violation_detected
        )
        
        return ExplanationResponse(
            report=violation_report,
            evidence_details={
                "transaction_evidence": transaction_evidence,
                "obligation_evidence": obligation_evidence,
                "similar_cases_count": len(similar_cases),
                "regulatory_references_count": len(regulatory_context)
            },
            metadata={
                "processing_time_ms": 0,  # Would be calculated in production
                "model_version": "gemini-1.5-flash",
                "evidence_sources": ["transaction", "obligation", "policy", "regulatory"]
            },
            success=True,
            errors=[]
        )
        
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/violations/{violation_id}")
async def get_violation_report(violation_id: str, db: Session = Depends(get_db)):
    """Retrieve existing violation report"""
    
    violation = db.query(Violation).filter(Violation.id == violation_id).first()
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    return {
        "violation_id": violation.id,
        "transaction_id": violation.transaction_id,
        "obligation_id": violation.obligation_id,
        "violation_type": violation.violation_type,
        "confidence": violation.confidence,
        "risk_level": violation.risk_level,
        "reasoning": violation.reasoning,
        "recommended_action": violation.recommended_action,
        "evidence_ids": violation.evidence_ids,
        "review_status": violation.review_status,
        "llm_explanation": violation.llm_explanation,
        "detected_at": violation.detected_at,
        "reviewed_at": violation.reviewed_at
    }


@app.put("/violations/{violation_id}/review")
async def review_violation(
    violation_id: str,
    review_status: str,  # approved, rejected
    review_notes: Optional[str] = None,
    reviewer_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Review and approve/reject violation"""
    
    if review_status not in ["approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Review status must be 'approved' or 'rejected'")
    
    violation = db.query(Violation).filter(Violation.id == violation_id).first()
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    violation.review_status = review_status
    violation.review_notes = review_notes
    violation.reviewer_id = reviewer_id
    violation.reviewed_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": f"Violation {review_status} successfully", "violation_id": violation_id}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "raggenerator", "timestamp": datetime.utcnow().isoformat()}


async def log_explanation_audit(violation_id: str, transaction_id: str, 
                              obligation_id: str, violation_detected: bool):
    """Log explanation generation audit event"""
    try:
        db = SessionLocal()
        
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            event_type="violation_explanation",
            entity_type="violation",
            entity_id=violation_id,
            event_data={
                "transaction_id": transaction_id,
                "obligation_id": obligation_id,
                "violation_detected": violation_detected
            }
        )
        
        db.add(audit_log)
        db.commit()
        db.close()
        
    except Exception as e:
        logger.error(f"Failed to log explanation audit: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)