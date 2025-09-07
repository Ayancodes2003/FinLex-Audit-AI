"""
Policy Obligation Extractor Service

FastAPI service for extracting structured obligations from policy documents
using Gemini-Flash with few-shot examples from ObliQa dataset.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import uuid
import logging
import asyncio
import re
from sqlalchemy.orm import Session

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_session_maker, PolicyDocument, Obligation, AuditLog
from gemini_client import get_gemini_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinLex Obligation Extractor Service",
    description="Extract structured obligations from policy documents using LLM",
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


class PolicyInput(BaseModel):
    """Input schema for policy document"""
    title: str = Field(..., description="Policy document title")
    content: str = Field(..., description="Policy document content")
    source: Optional[str] = Field(default="manual", description="Source identifier")
    jurisdiction: Optional[str] = Field(default="US", description="Applicable jurisdiction")
    effective_date: Optional[datetime] = Field(default=None, description="Effective date")
    expiry_date: Optional[datetime] = Field(default=None, description="Expiry date")


class ExtractionRequest(BaseModel):
    """Request for obligation extraction"""
    policy: PolicyInput
    use_few_shot: bool = Field(default=True, description="Use ObliQa few-shot examples")
    max_obligations: int = Field(default=50, description="Maximum obligations to extract")


class ObligationOutput(BaseModel):
    """Output schema for extracted obligation"""
    id: str
    actor: str
    action: str
    type: str
    condition: Optional[str]
    jurisdiction: Optional[str]
    temporal_scope: Optional[str]
    confidence: float
    source_clause: str


class ExtractionResponse(BaseModel):
    """Response for obligation extraction"""
    policy_id: str
    obligations: List[ObligationOutput]
    extraction_metadata: Dict[str, Any]
    success: bool
    errors: List[str]


class PolicyClauseSplitter:
    """Split policy documents into analyzable clauses"""
    
    @staticmethod
    def split_into_clauses(content: str) -> List[str]:
        """Split policy content into individual clauses"""
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Split by common sentence/clause terminators
        clause_patterns = [
            r'\.\s+(?=[A-Z])',  # Period followed by space and capital letter
            r';\s+(?=[A-Z])',   # Semicolon followed by space and capital letter
            r'\n\s*(?=\d+\.)',  # Numbered items
            r'\n\s*(?=[A-Z])',  # New lines with capital letters
        ]
        
        clauses = [content]
        for pattern in clause_patterns:
            new_clauses = []
            for clause in clauses:
                new_clauses.extend(re.split(pattern, clause))
            clauses = new_clauses
        
        # Filter and clean clauses
        filtered_clauses = []
        for clause in clauses:
            clause = clause.strip()
            # Keep clauses that are substantial and likely contain obligations
            if (len(clause) > 20 and 
                any(keyword in clause.lower() for keyword in 
                    ['must', 'shall', 'required', 'prohibited', 'comply', 'report', 'maintain'])):
                filtered_clauses.append(clause)
        
        return filtered_clauses[:50]  # Limit to 50 clauses


class ObliQaExampleProvider:
    """Provide few-shot examples from ObliQa dataset"""
    
    # Mock ObliQa examples (in production, load from actual dataset)
    OBLIGATION_EXAMPLES = [
        {
            "policy": "Financial institutions must report currency transactions exceeding $10,000 to FinCEN within 15 days.",
            "obligation": {
                "actor": "financial institutions",
                "action": "report currency transactions exceeding $10,000 to FinCEN",
                "type": "requirement",
                "condition": "transaction amount > $10,000",
                "jurisdiction": "US",
                "temporal_scope": "within 15 days",
                "confidence": 0.95
            }
        },
        {
            "policy": "Banks are prohibited from processing transactions involving sanctioned entities without proper authorization.",
            "obligation": {
                "actor": "banks",
                "action": "process transactions involving sanctioned entities without authorization",
                "type": "prohibition",
                "condition": "involving sanctioned entities",
                "jurisdiction": "US",
                "temporal_scope": "ongoing",
                "confidence": 0.92
            }
        },
        {
            "policy": "Investment advisers shall maintain records of all client communications for a minimum of five years.",
            "obligation": {
                "actor": "investment advisers",
                "action": "maintain records of all client communications",
                "type": "requirement",
                "condition": "all client communications",
                "jurisdiction": "US",
                "temporal_scope": "minimum of five years",
                "confidence": 0.98
            }
        }
    ]
    
    @classmethod
    def get_relevant_examples(cls, policy_content: str, max_examples: int = 3) -> List[Dict]:
        """Get relevant few-shot examples based on policy content"""
        
        # Simple keyword matching for example selection
        # In production, use semantic similarity with embeddings
        policy_lower = policy_content.lower()
        
        scored_examples = []
        for example in cls.OBLIGATION_EXAMPLES:
            score = 0
            example_text = example['policy'].lower()
            
            # Score based on keyword overlap
            common_keywords = ['report', 'transaction', 'maintain', 'record', 'prohibited']
            for keyword in common_keywords:
                if keyword in policy_lower and keyword in example_text:
                    score += 1
            
            scored_examples.append((score, example))
        
        # Sort by score and return top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example[1] for example in scored_examples[:max_examples]]


class ObligationExtractor:
    """Core obligation extraction logic"""
    
    def __init__(self):
        self.gemini_client = get_gemini_client()
        self.clause_splitter = PolicyClauseSplitter()
        self.example_provider = ObliQaExampleProvider()
    
    async def extract_obligations(self, policy: PolicyInput, use_few_shot: bool = True) -> Dict[str, Any]:
        """Extract obligations from policy document"""
        
        logger.info(f"Extracting obligations from policy: {policy.title}")
        
        # Split policy into clauses
        clauses = self.clause_splitter.split_into_clauses(policy.content)
        logger.info(f"Split policy into {len(clauses)} clauses")
        
        all_obligations = []
        extraction_errors = []
        
        # Get few-shot examples if requested
        few_shot_examples = []
        if use_few_shot:
            few_shot_examples = self.example_provider.get_relevant_examples(policy.content)
        
        # Process clauses in batches to avoid token limits
        batch_size = 5
        for i in range(0, len(clauses), batch_size):
            batch_clauses = clauses[i:i+batch_size]
            batch_text = "\n\n".join(f"Clause {i+j+1}: {clause}" 
                                   for j, clause in enumerate(batch_clauses))
            
            try:
                # Call Gemini for obligation extraction
                response = await self.gemini_client.extract_obligations(
                    policy_text=batch_text,
                    few_shot_examples=few_shot_examples
                )
                
                # Process extracted obligations
                if 'obligations' in response.json_output:
                    for obl_data in response.json_output['obligations']:
                        # Validate and enrich obligation data
                        obligation = self._process_obligation_data(
                            obl_data, policy, batch_clauses
                        )
                        if obligation:
                            all_obligations.append(obligation)
                
            except Exception as e:
                error_msg = f"Failed to extract from batch {i//batch_size + 1}: {str(e)}"
                extraction_errors.append(error_msg)
                logger.error(error_msg)
        
        return {
            'obligations': all_obligations,
            'clause_count': len(clauses),
            'batch_count': len(range(0, len(clauses), batch_size)),
            'errors': extraction_errors,
            'few_shot_used': use_few_shot
        }
    
    def _process_obligation_data(self, obl_data: Dict, policy: PolicyInput, 
                               source_clauses: List[str]) -> Optional[Dict]:
        """Process and validate extracted obligation data"""
        
        try:
            # Generate unique obligation ID
            obligation_id = self._generate_obligation_id(obl_data, policy.title)
            
            # Find most relevant source clause
            source_clause = self._find_source_clause(obl_data, source_clauses)
            
            return {
                'id': obligation_id,
                'actor': obl_data.get('actor', '').strip(),
                'action': obl_data.get('action', '').strip(),
                'type': obl_data.get('type', 'requirement').lower(),
                'condition': obl_data.get('condition', '').strip() or None,
                'jurisdiction': obl_data.get('jurisdiction') or policy.jurisdiction,
                'temporal_scope': obl_data.get('temporal_scope', '').strip() or None,
                'confidence': float(obl_data.get('confidence', 0.5)),
                'source_clause': source_clause,
                'extraction_model': 'gemini-1.5-flash'
            }
            
        except Exception as e:
            logger.warning(f"Failed to process obligation data: {e}")
            return None
    
    def _generate_obligation_id(self, obl_data: Dict, policy_title: str) -> str:
        """Generate unique obligation ID"""
        content = f"{policy_title}_{obl_data.get('actor', '')}_{obl_data.get('action', '')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _find_source_clause(self, obl_data: Dict, clauses: List[str]) -> str:
        """Find the most relevant source clause for the obligation"""
        
        action = obl_data.get('action', '').lower()
        actor = obl_data.get('actor', '').lower()
        
        # Simple keyword matching to find source clause
        best_clause = clauses[0] if clauses else "Unknown source"
        best_score = 0
        
        for clause in clauses:
            clause_lower = clause.lower()
            score = 0
            
            # Score based on keyword presence
            if action and any(word in clause_lower for word in action.split()[:3]):
                score += 2
            if actor and any(word in clause_lower for word in actor.split()[:2]):
                score += 1
                
            if score > best_score:
                best_score = score
                best_clause = clause
        
        return best_clause[:500]  # Truncate for storage


extractor = ObligationExtractor()


@app.post("/extract", response_model=ExtractionResponse)
async def extract_obligations(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Extract obligations from policy document"""
    
    logger.info(f"Processing extraction request for policy: {request.policy.title}")
    
    try:
        # Generate policy document ID
        policy_content_hash = hashlib.sha256(request.policy.content.encode()).hexdigest()
        policy_id = f"pol_{policy_content_hash[:16]}"
        
        # Check if policy already processed
        existing_policy = db.query(PolicyDocument).filter(
            PolicyDocument.document_hash == policy_content_hash
        ).first()
        
        if existing_policy and existing_policy.is_processed:
            # Return existing obligations
            obligations = db.query(Obligation).filter(
                Obligation.policy_document_id == existing_policy.id
            ).all()
            
            obligation_outputs = [
                ObligationOutput(
                    id=obl.id,
                    actor=obl.actor,
                    action=obl.action,
                    type=obl.type,
                    condition=obl.condition,
                    jurisdiction=obl.jurisdiction,
                    temporal_scope=obl.temporal_scope,
                    confidence=obl.confidence,
                    source_clause=obl.source_clause
                )
                for obl in obligations
            ]
            
            return ExtractionResponse(
                policy_id=existing_policy.id,
                obligations=obligation_outputs,
                extraction_metadata={"cached": True},
                success=True,
                errors=[]
            )
        
        # Create new policy document record
        policy_doc = PolicyDocument(
            id=policy_id,
            title=request.policy.title,
            content=request.policy.content,
            document_hash=policy_content_hash,
            source=request.policy.source,
            jurisdiction=request.policy.jurisdiction,
            effective_date=request.policy.effective_date,
            expiry_date=request.policy.expiry_date,
            is_processed=False
        )
        
        db.add(policy_doc)
        db.flush()  # Get the ID without committing
        
        # Extract obligations
        extraction_result = await extractor.extract_obligations(
            request.policy, 
            request.use_few_shot
        )
        
        # Save obligations to database
        obligation_records = []
        obligation_outputs = []
        
        for obl_data in extraction_result['obligations'][:request.max_obligations]:
            obligation = Obligation(
                id=obl_data['id'],
                policy_document_id=policy_doc.id,
                actor=obl_data['actor'],
                action=obl_data['action'],
                type=obl_data['type'],
                condition=obl_data['condition'],
                jurisdiction=obl_data['jurisdiction'],
                temporal_scope=obl_data['temporal_scope'],
                confidence=obl_data['confidence'],
                extraction_model=obl_data['extraction_model'],
                source_clause=obl_data['source_clause']
            )
            
            obligation_records.append(obligation)
            obligation_outputs.append(ObligationOutput(**obl_data))
        
        # Save to database
        db.add_all(obligation_records)
        policy_doc.is_processed = True
        db.commit()
        
        # Log audit event
        background_tasks.add_task(
            log_extraction_audit,
            policy_id=policy_doc.id,
            obligation_count=len(obligation_records),
            extraction_metadata=extraction_result
        )
        
        logger.info(f"Successfully extracted {len(obligation_records)} obligations")
        
        return ExtractionResponse(
            policy_id=policy_doc.id,
            obligations=obligation_outputs,
            extraction_metadata={
                "clause_count": extraction_result['clause_count'],
                "batch_count": extraction_result['batch_count'],
                "few_shot_used": extraction_result['few_shot_used']
            },
            success=True,
            errors=extraction_result['errors']
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@app.post("/extract/file")
async def extract_from_file(
    file: UploadFile = File(...),
    title: str = None,
    jurisdiction: str = "US",
    use_few_shot: bool = True,
    db: Session = Depends(get_db)
):
    """Extract obligations from uploaded policy file"""
    
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only .txt and .md files supported")
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        policy_input = PolicyInput(
            title=title or file.filename,
            content=content_str,
            source=f"file_upload_{file.filename}",
            jurisdiction=jurisdiction
        )
        
        request = ExtractionRequest(
            policy=policy_input,
            use_few_shot=use_few_shot
        )
        
        return await extract_obligations(request, BackgroundTasks(), db)
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@app.get("/policies/{policy_id}/obligations")
async def get_policy_obligations(policy_id: str, db: Session = Depends(get_db)):
    """Get all obligations for a specific policy"""
    
    obligations = db.query(Obligation).filter(
        Obligation.policy_document_id == policy_id
    ).all()
    
    if not obligations:
        raise HTTPException(status_code=404, detail="Policy or obligations not found")
    
    return [
        {
            "id": obl.id,
            "actor": obl.actor,
            "action": obl.action,
            "type": obl.type,
            "condition": obl.condition,
            "jurisdiction": obl.jurisdiction,
            "temporal_scope": obl.temporal_scope,
            "confidence": obl.confidence,
            "source_clause": obl.source_clause,
            "created_at": obl.created_at
        }
        for obl in obligations
    ]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "extractor", "timestamp": datetime.utcnow().isoformat()}


async def log_extraction_audit(policy_id: str, obligation_count: int, extraction_metadata: Dict):
    """Log extraction audit event"""
    try:
        db = SessionLocal()
        
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            event_type="obligation_extraction",
            entity_type="policy",
            entity_id=policy_id,
            event_data={
                "obligation_count": obligation_count,
                "extraction_metadata": extraction_metadata
            }
        )
        
        db.add(audit_log)
        db.commit()
        db.close()
        
    except Exception as e:
        logger.error(f"Failed to log extraction audit: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)