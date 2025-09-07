"""
Main API Service for FinLex Audit AI

FastAPI service that orchestrates all compliance analysis services
and provides a unified API for the Streamlit frontend.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncio
import httpx
import os
from sqlalchemy.orm import Session

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_session_maker, Transaction, Violation, PolicyDocument, Obligation, init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinLex Audit AI API",
    description="Unified API for AI-powered financial compliance analysis",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service endpoints
SERVICE_URLS = {
    "ingest": os.getenv("INGEST_SERVICE_URL", "http://localhost:8001"),
    "extractor": os.getenv("EXTRACTOR_SERVICE_URL", "http://localhost:8002"),
    "matcher": os.getenv("MATCHER_SERVICE_URL", "http://localhost:8003"),
    "raggenerator": os.getenv("RAGGENERATOR_SERVICE_URL", "http://localhost:8004")
}

# Database dependency
SessionLocal = get_session_maker()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ComplianceScanRequest(BaseModel):
    """Request for full compliance scan"""
    transaction_ids: List[str] = Field(..., description="Transaction IDs to scan")
    jurisdiction: str = Field(default="US", description="Jurisdiction for compliance rules")
    generate_explanations: bool = Field(default=True, description="Generate detailed explanations")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")


class ComplianceScanResult(BaseModel):
    """Result of compliance scan"""
    scan_id: str
    transaction_count: int
    violation_count: int
    violations: List[Dict[str, Any]]
    summary: Dict[str, Any]
    processing_time_seconds: float
    success: bool
    errors: List[str]


class ComplianceOrchestrator:
    """Orchestrate the complete compliance analysis pipeline"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def full_compliance_scan(self, request: ComplianceScanRequest) -> Dict[str, Any]:
        """Run complete compliance scan pipeline"""
        
        start_time = datetime.utcnow()
        scan_id = f"scan_{int(start_time.timestamp())}"
        
        logger.info(f"Starting compliance scan {scan_id} for {len(request.transaction_ids)} transactions")
        
        try:
            # Step 1: Run compliance matching
            violations = await self._run_compliance_matching(
                request.transaction_ids,
                request.jurisdiction,
                request.confidence_threshold
            )
            
            # Step 2: Generate explanations for violations (if requested)
            explained_violations = []
            if request.generate_explanations and violations:
                explained_violations = await self._generate_violation_explanations(violations)
            else:
                explained_violations = violations
            
            # Step 3: Calculate summary statistics
            summary = self._calculate_scan_summary(
                len(request.transaction_ids),
                explained_violations
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "scan_id": scan_id,
                "transaction_count": len(request.transaction_ids),
                "violation_count": len(explained_violations),
                "violations": explained_violations,
                "summary": summary,
                "processing_time_seconds": processing_time,
                "success": True,
                "errors": []
            }
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Compliance scan {scan_id} failed: {e}")
            
            return {
                "scan_id": scan_id,
                "transaction_count": len(request.transaction_ids),
                "violation_count": 0,
                "violations": [],
                "summary": {"error": str(e)},
                "processing_time_seconds": processing_time,
                "success": False,
                "errors": [str(e)]
            }
    
    async def _run_compliance_matching(self, transaction_ids: List[str], 
                                     jurisdiction: str, 
                                     confidence_threshold: float) -> List[Dict[str, Any]]:
        """Run compliance matching service"""
        
        match_request = {
            "transaction_ids": transaction_ids,
            "jurisdiction": jurisdiction,
            "rule_types": ["threshold", "semantic"],
            "confidence_threshold": confidence_threshold
        }
        
        async with self.http_client as client:
            response = await client.post(
                f"{SERVICE_URLS['matcher']}/match",
                json=match_request
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Matching service error: {response.text}"
                )
            
            result = response.json()
            return result.get("matches", [])
    
    async def _generate_violation_explanations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate explanations for detected violations"""
        
        explained_violations = []
        
        # Process violations in batches to avoid overwhelming the RAG service
        batch_size = 5
        for i in range(0, len(violations), batch_size):
            batch = violations[i:i+batch_size]
            
            # Process batch concurrently
            tasks = []
            for violation in batch:
                task = self._explain_single_violation(violation)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for violation, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to explain violation: {result}")
                    # Use original violation without explanation
                    explained_violations.append(violation)
                else:
                    explained_violations.append(result)
        
        return explained_violations
    
    async def _explain_single_violation(self, violation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for a single violation"""
        
        explanation_request = {
            "transaction_id": violation["transaction_id"],
            "obligation_id": violation["obligation_id"],
            "violation_type": violation["violation_type"],
            "confidence": violation["confidence"]
        }
        
        try:
            async with self.http_client as client:
                response = await client.post(
                    f"{SERVICE_URLS['raggenerator']}/generate",
                    json=explanation_request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    report = result.get("report", {})
                    
                    # Merge explanation into violation
                    violation.update({
                        "violation_id": report.get("violation_id"),
                        "human_explanation": report.get("human_explanation"),
                        "regulatory_references": report.get("regulatory_references", []),
                        "next_steps": report.get("next_steps", []),
                        "detailed_reasoning": report.get("reasoning")
                    })
                
                return violation
                
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            return violation
    
    def _calculate_scan_summary(self, transaction_count: int, 
                              violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate scan summary statistics"""
        
        if not violations:
            return {
                "total_transactions": transaction_count,
                "clean_transactions": transaction_count,
                "violation_rate": 0.0,
                "risk_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
                "violation_types": {},
                "avg_confidence": 0.0
            }
        
        # Count by risk level
        risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        violation_types = {}
        confidences = []
        
        for violation in violations:
            risk_level = violation.get("risk_level", "medium")
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            
            violation_type = violation.get("violation_type", "unknown")
            violation_types[violation_type] = violation_types.get(violation_type, 0) + 1
            
            confidences.append(violation.get("confidence", 0.5))
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        violation_rate = len(violations) / transaction_count if transaction_count > 0 else 0.0
        
        return {
            "total_transactions": transaction_count,
            "clean_transactions": transaction_count - len(violations),
            "violation_rate": violation_rate,
            "risk_distribution": risk_distribution,
            "violation_types": violation_types,
            "avg_confidence": avg_confidence,
            "high_risk_violations": risk_distribution.get("high", 0) + risk_distribution.get("critical", 0)
        }


orchestrator = ComplianceOrchestrator()


@app.post("/scan", response_model=ComplianceScanResult)
async def run_compliance_scan(
    request: ComplianceScanRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Run complete compliance scan"""
    
    # Validate that transactions exist
    existing_transactions = db.query(Transaction.id).filter(
        Transaction.id.in_(request.transaction_ids)
    ).all()
    
    existing_ids = [tx.id for tx in existing_transactions]
    missing_ids = set(request.transaction_ids) - set(existing_ids)
    
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Transactions not found: {list(missing_ids)[:5]}"  # Show first 5 missing
        )
    
    # Run compliance scan
    result = await orchestrator.full_compliance_scan(request)
    
    return ComplianceScanResult(**result)


@app.post("/transactions/upload")
async def upload_transactions(
    file: UploadFile = File(...),
    source: str = "csv_upload",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Upload transaction CSV file"""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV and convert to transaction format
        import csv
        import io
        
        reader = csv.DictReader(io.StringIO(csv_content))
        transactions = []
        
        for row in reader:
            # Map CSV columns to transaction schema
            transaction = {
                "step": int(row.get("step", 1)),
                "type": row.get("type", "PAYMENT").upper(),
                "amount": float(row.get("amount", 0)),
                "currency": row.get("currency", "USD"),
                "nameOrig": row.get("nameOrig", f"account_{len(transactions)}"),
                "oldbalanceOrg": float(row.get("oldbalanceOrg", 0)),
                "newbalanceOrig": float(row.get("newbalanceOrig", 0)),
                "nameDest": row.get("nameDest", f"dest_{len(transactions)}"),
                "oldbalanceDest": float(row.get("oldbalanceDest", 0)),
                "newbalanceDest": float(row.get("newbalanceDest", 0)),
                "isFraud": str(row.get("isFraud", "false")).lower() == "true",
                "isFlaggedMerchant": str(row.get("isFlaggedMerchant", "false")).lower() == "true"
            }
            transactions.append(transaction)
        
        # Forward to ingest service
        batch_data = {
            "transactions": transactions,
            "source": source
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SERVICE_URLS['ingest']}/transactions",
                json=batch_data
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ingest service error: {response.text}"
                )
            
            return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing error: {str(e)}")


@app.post("/policies/upload")
async def upload_policy(
    file: UploadFile = File(...),
    title: str = None,
    jurisdiction: str = "US",
    use_few_shot: bool = True
):
    """Upload and process policy document"""
    
    if not file.filename.endswith(('.txt', '.md', '.pdf')):
        raise HTTPException(status_code=400, detail="Supported formats: .txt, .md, .pdf")
    
    try:
        # Read file content
        content = await file.read()
        
        if file.filename.endswith('.pdf'):
            # TODO: Add PDF processing with PyPDF2 or similar
            raise HTTPException(status_code=400, detail="PDF processing not yet implemented")
        
        content_str = content.decode('utf-8')
        
        # Forward to extractor service
        policy_data = {
            "title": title or file.filename,
            "content": content_str,
            "source": f"file_upload_{file.filename}",
            "jurisdiction": jurisdiction
        }
        
        extract_request = {
            "policy": policy_data,
            "use_few_shot": use_few_shot
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SERVICE_URLS['extractor']}/extract",
                json=extract_request
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Extractor service error: {response.text}"
                )
            
            return response.json()
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy processing error: {str(e)}")


@app.get("/dashboard/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    
    # Count entities
    transaction_count = db.query(Transaction).count()
    policy_count = db.query(PolicyDocument).count()
    obligation_count = db.query(Obligation).count()
    violation_count = db.query(Violation).count()
    
    # Recent violations
    recent_violations = db.query(Violation).filter(
        Violation.detected_at >= datetime.utcnow() - timedelta(days=7)
    ).count()
    
    # Pending reviews
    pending_reviews = db.query(Violation).filter(
        Violation.review_status == "pending"
    ).count()
    
    return {
        "total_transactions": transaction_count,
        "total_policies": policy_count,
        "total_obligations": obligation_count,
        "total_violations": violation_count,
        "recent_violations": recent_violations,
        "pending_reviews": pending_reviews,
        "system_status": "operational"
    }


@app.get("/violations/recent")
async def get_recent_violations(
    limit: int = 10,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get recent violations for dashboard"""
    
    query = db.query(Violation).order_by(Violation.detected_at.desc())
    
    if status:
        query = query.filter(Violation.review_status == status)
    
    violations = query.limit(limit).all()
    
    return [
        {
            "id": v.id,
            "transaction_id": v.transaction_id,
            "violation_type": v.violation_type,
            "risk_level": v.risk_level,
            "confidence": v.confidence,
            "review_status": v.review_status,
            "detected_at": v.detected_at,
            "recommended_action": v.recommended_action
        }
        for v in violations
    ]


@app.post("/violations/{violation_id}/review")
async def review_violation(
    violation_id: str,
    action: str,  # approve, reject
    notes: Optional[str] = None,
    reviewer_id: Optional[str] = None
):
    """Review violation (forward to RAG service)"""
    
    review_data = {
        "review_status": "approved" if action == "approve" else "rejected",
        "review_notes": notes,
        "reviewer_id": reviewer_id
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{SERVICE_URLS['raggenerator']}/violations/{violation_id}/review",
            params=review_data
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Review service error: {response.text}"
            )
        
        return response.json()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    # Check service health
    service_health = {}
    
    for service_name, service_url in SERVICE_URLS.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service_url}/health")
                service_health[service_name] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            service_health[service_name] = "unreachable"
    
    overall_health = "healthy" if all(status == "healthy" for status in service_health.values()) else "degraded"
    
    return {
        "status": overall_health,
        "services": service_health,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting FinLex Audit AI API service")
    
    # Initialize database
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)