"""
Transaction Ingest Service

FastAPI service for ingesting and normalizing Paysim1-like transaction data.
Handles normalization, PII hashing, and derived feature computation.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import hashlib
import uuid
import asyncio
import logging
from sqlalchemy.orm import Session

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_session_maker, Transaction, AuditLog
from gemini_client import get_gemini_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinLex Transaction Ingest Service",
    description="Ingest and normalize financial transaction data for compliance analysis",
    version="1.0.0"
)

# Currency conversion rates (mock - in production, use real-time FX API)
CURRENCY_RATES = {
    'USD': 1.0,
    'EUR': 1.08,
    'GBP': 1.25,
    'JPY': 0.007,
    'CAD': 0.74,
    'AUD': 0.67
}


class TransactionInput(BaseModel):
    """Input schema for transaction ingestion"""
    step: int = Field(..., description="Transaction step/sequence number")
    type: str = Field(..., description="Transaction type (PAYMENT, TRANSFER, etc.)")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Transaction currency")
    nameOrig: str = Field(..., description="Origin account name")
    oldbalanceOrg: Optional[float] = Field(default=0.0, description="Origin balance before")
    newbalanceOrig: Optional[float] = Field(default=0.0, description="Origin balance after")
    nameDest: str = Field(..., description="Destination account name")
    oldbalanceDest: Optional[float] = Field(default=0.0, description="Destination balance before")
    newbalanceDest: Optional[float] = Field(default=0.0, description="Destination balance after")
    isFraud: Optional[bool] = Field(default=False, description="Known fraud flag")
    isFlaggedMerchant: Optional[bool] = Field(default=False, description="Flagged merchant indicator")
    timestamp: Optional[datetime] = Field(default=None, description="Transaction timestamp")
    
    @validator('type')
    def validate_transaction_type(cls, v):
        valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        if v.upper() not in valid_types:
            raise ValueError(f'Transaction type must be one of {valid_types}')
        return v.upper()
    
    @validator('currency')
    def validate_currency(cls, v):
        if v.upper() not in CURRENCY_RATES:
            raise ValueError(f'Unsupported currency: {v}. Supported: {list(CURRENCY_RATES.keys())}')
        return v.upper()
    
    @validator('timestamp', pre=True)
    def validate_timestamp(cls, v):
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class TransactionBatch(BaseModel):
    """Batch of transactions for bulk ingestion"""
    transactions: List[TransactionInput]
    source: Optional[str] = Field(default="api", description="Data source identifier")


class IngestResponse(BaseModel):
    """Response for transaction ingestion"""
    success: bool
    processed_count: int
    failed_count: int
    transaction_ids: List[str]
    errors: List[str]


# Database dependency
SessionLocal = get_session_maker()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TransactionProcessor:
    """Core transaction processing logic"""
    
    @staticmethod
    def hash_pii(value: str) -> str:
        """Hash PII data using SHA-256"""
        return hashlib.sha256(value.encode()).hexdigest()
    
    @staticmethod
    def normalize_currency(amount: float, currency: str) -> float:
        """Convert amount to USD using exchange rates"""
        rate = CURRENCY_RATES.get(currency.upper(), 1.0)
        return amount * rate
    
    @staticmethod
    def normalize_timestamp(timestamp: datetime) -> datetime:
        """Ensure timestamp is in UTC"""
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)
    
    @staticmethod
    def generate_transaction_id(transaction_data: Dict[str, Any]) -> str:
        """Generate deterministic transaction ID"""
        # Create hash from key transaction attributes
        hash_input = f"{transaction_data['step']}_{transaction_data['nameOrig']}_{transaction_data['nameDest']}_{transaction_data['amount']}_{transaction_data['timestamp']}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    async def compute_derived_features(self, db: Session, transaction: Transaction) -> Dict[str, float]:
        """Compute rolling window aggregations for risk features"""
        # In production, this would use efficient windowing queries
        # For now, return mock values
        return {
            'amount_1d_total': transaction.amount_usd * 1.2,
            'amount_30d_total': transaction.amount_usd * 15.8,
            'transaction_count_1d': 3,
            'transaction_count_30d': 47
        }
    
    async def process_transaction(self, db: Session, tx_input: TransactionInput) -> Transaction:
        """Process and normalize a single transaction"""
        
        # Normalize timestamp to UTC
        normalized_timestamp = self.normalize_timestamp(
            tx_input.timestamp or datetime.now(timezone.utc)
        )
        
        # Convert amount to USD
        amount_usd = self.normalize_currency(tx_input.amount, tx_input.currency)
        
        # Hash PII fields
        name_orig_hash = self.hash_pii(tx_input.nameOrig)
        name_dest_hash = self.hash_pii(tx_input.nameDest)
        
        # Generate transaction ID
        tx_id = self.generate_transaction_id({
            'step': tx_input.step,
            'nameOrig': tx_input.nameOrig,
            'nameDest': tx_input.nameDest,
            'amount': amount_usd,
            'timestamp': normalized_timestamp.isoformat()
        })
        
        # Create transaction record
        transaction = Transaction(
            id=tx_id,
            step=tx_input.step,
            type=tx_input.type,
            amount_usd=amount_usd,
            name_orig_hash=name_orig_hash,
            old_balance_orig=tx_input.oldbalanceOrg,
            new_balance_orig=tx_input.newbalanceOrig,
            name_dest_hash=name_dest_hash,
            old_balance_dest=tx_input.oldbalanceDest,
            new_balance_dest=tx_input.newbalanceDest,
            is_fraud=tx_input.isFraud,
            is_flagged_merchant=tx_input.isFlaggedMerchant,
            timestamp_utc=normalized_timestamp
        )
        
        # Compute derived features
        derived_features = await self.compute_derived_features(db, transaction)
        transaction.amount_1d_total = derived_features['amount_1d_total']
        transaction.amount_30d_total = derived_features['amount_30d_total']
        transaction.transaction_count_1d = derived_features['transaction_count_1d']
        transaction.transaction_count_30d = derived_features['transaction_count_30d']
        
        return transaction


processor = TransactionProcessor()


@app.post("/transactions", response_model=IngestResponse)
async def ingest_transactions(
    batch: TransactionBatch,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Ingest batch of transactions"""
    
    logger.info(f"Processing batch of {len(batch.transactions)} transactions from source: {batch.source}")
    
    processed_transactions = []
    errors = []
    
    for idx, tx_input in enumerate(batch.transactions):
        try:
            # Process transaction
            transaction = await processor.process_transaction(db, tx_input)
            
            # Check for duplicates
            existing = db.query(Transaction).filter(Transaction.id == transaction.id).first()
            if existing:
                errors.append(f"Transaction {idx}: Duplicate transaction ID {transaction.id}")
                continue
            
            # Save to database
            db.add(transaction)
            processed_transactions.append(transaction)
            
        except Exception as e:
            errors.append(f"Transaction {idx}: {str(e)}")
            logger.error(f"Failed to process transaction {idx}: {e}")
    
    # Commit successful transactions
    try:
        db.commit()
        transaction_ids = [tx.id for tx in processed_transactions]
        
        # Log audit trail
        background_tasks.add_task(
            log_audit_event,
            event_type="batch_ingest",
            entity_type="transaction",
            event_data={
                "source": batch.source,
                "processed_count": len(processed_transactions),
                "failed_count": len(errors),
                "transaction_ids": transaction_ids
            }
        )
        
        logger.info(f"Successfully processed {len(processed_transactions)} transactions")
        
        return IngestResponse(
            success=True,
            processed_count=len(processed_transactions),
            failed_count=len(errors),
            transaction_ids=transaction_ids,
            errors=errors
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Database commit failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/transactions/{transaction_id}")
async def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    """Retrieve a specific transaction"""
    
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {
        "id": transaction.id,
        "step": transaction.step,
        "type": transaction.type,
        "amount_usd": transaction.amount_usd,
        "timestamp_utc": transaction.timestamp_utc,
        "is_fraud": transaction.is_fraud,
        "derived_features": {
            "amount_1d_total": transaction.amount_1d_total,
            "amount_30d_total": transaction.amount_30d_total,
            "transaction_count_1d": transaction.transaction_count_1d,
            "transaction_count_30d": transaction.transaction_count_30d
        },
        "created_at": transaction.created_at
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ingest", "timestamp": datetime.utcnow().isoformat()}


async def log_audit_event(event_type: str, entity_type: str, event_data: Dict[str, Any]):
    """Log audit event in background"""
    try:
        db = SessionLocal()
        
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            event_type=event_type,
            entity_type=entity_type,
            event_data=event_data,
            data_hash=hashlib.sha256(str(event_data).encode()).hexdigest()
        )
        
        db.add(audit_log)
        db.commit()
        db.close()
        
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)