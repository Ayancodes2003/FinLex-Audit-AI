"""
Database models for FinLex Audit AI

SQLAlchemy ORM models for transactions, obligations, violations, and audit trails.
"""

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, Boolean, 
    Text, JSON, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import os

Base = declarative_base()


class Transaction(Base):
    """Normalized transaction records from Paysim1-like data"""
    __tablename__ = 'transactions'
    
    id = Column(String(64), primary_key=True)  # Hashed transaction ID
    step = Column(Integer, nullable=False)
    type = Column(String(20), nullable=False)  # PAYMENT, TRANSFER, etc.
    amount_usd = Column(Float, nullable=False)  # Normalized to USD
    name_orig_hash = Column(String(64), nullable=False)  # Hashed PII
    old_balance_orig = Column(Float)
    new_balance_orig = Column(Float)
    name_dest_hash = Column(String(64), nullable=False)  # Hashed PII
    old_balance_dest = Column(Float)
    new_balance_dest = Column(Float)
    is_fraud = Column(Boolean, default=False)
    is_flagged_merchant = Column(Boolean, default=False)
    
    # Derived features
    amount_1d_total = Column(Float, default=0.0)  # 1-day rolling sum
    amount_30d_total = Column(Float, default=0.0)  # 30-day rolling sum
    transaction_count_1d = Column(Integer, default=0)
    transaction_count_30d = Column(Integer, default=0)
    
    # Metadata
    timestamp_utc = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    violations = relationship("Violation", back_populates="transaction")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_transaction_amount', 'amount_usd'),
        Index('idx_transaction_timestamp', 'timestamp_utc'),
        Index('idx_transaction_type', 'type'),
        Index('idx_transaction_names', 'name_orig_hash', 'name_dest_hash'),
    )


class PolicyDocument(Base):
    """Policy documents and their processing status"""
    __tablename__ = 'policy_documents'
    
    id = Column(String(64), primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    document_hash = Column(String(64), nullable=False, unique=True)
    source = Column(String(100))  # C3PA, manual upload, etc.
    jurisdiction = Column(String(50))
    effective_date = Column(DateTime)
    expiry_date = Column(DateTime)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    obligations = relationship("Obligation", back_populates="policy_document")


class Obligation(Base):
    """Structured obligations extracted from policies"""
    __tablename__ = 'obligations'
    
    id = Column(String(64), primary_key=True)
    policy_document_id = Column(String(64), ForeignKey('policy_documents.id'), nullable=False)
    
    # Structured obligation fields
    actor = Column(String(255), nullable=False)
    action = Column(Text, nullable=False)
    type = Column(String(50), nullable=False)  # requirement, prohibition, condition
    condition = Column(Text)
    jurisdiction = Column(String(50))
    temporal_scope = Column(String(255))
    
    # Extraction metadata
    confidence = Column(Float, nullable=False)
    extraction_model = Column(String(50))
    source_clause = Column(Text)  # Original policy text
    
    # Vector embedding reference
    embedding_id = Column(String(64))  # Reference to FAISS vector
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    policy_document = relationship("PolicyDocument", back_populates="obligations")
    violations = relationship("Violation", back_populates="obligation")
    
    # Indexes
    __table_args__ = (
        Index('idx_obligation_type', 'type'),
        Index('idx_obligation_jurisdiction', 'jurisdiction'),
        Index('idx_obligation_actor', 'actor'),
    )


class Violation(Base):
    """Detected compliance violations"""
    __tablename__ = 'violations'
    
    id = Column(String(64), primary_key=True)
    transaction_id = Column(String(64), ForeignKey('transactions.id'), nullable=False)
    obligation_id = Column(String(64), ForeignKey('obligations.id'), nullable=False)
    
    # Detection details
    violation_type = Column(String(50), nullable=False)  # threshold, semantic, etc.
    confidence = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # low, medium, high, critical
    
    # Analysis results
    reasoning = Column(Text, nullable=False)
    recommended_action = Column(String(50), nullable=False)
    evidence_ids = Column(JSON)  # List of evidence references
    
    # Gemini response metadata
    llm_response_hash = Column(String(64))
    llm_explanation = Column(Text)
    model_version = Column(String(50))
    
    # Review status
    review_status = Column(String(20), default='pending')  # pending, approved, rejected
    reviewer_id = Column(String(64))
    review_notes = Column(Text)
    reviewed_at = Column(DateTime)
    
    # Metadata
    detected_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    transaction = relationship("Transaction", back_populates="violations")
    obligation = relationship("Obligation", back_populates="violations")
    
    # Indexes
    __table_args__ = (
        Index('idx_violation_status', 'review_status'),
        Index('idx_violation_risk', 'risk_level'),
        Index('idx_violation_detected', 'detected_at'),
    )


class AuditLog(Base):
    """Comprehensive audit trail for compliance tracking"""
    __tablename__ = 'audit_logs'
    
    id = Column(String(64), primary_key=True)
    event_type = Column(String(50), nullable=False)  # llm_call, violation_detected, etc.
    entity_type = Column(String(50))  # transaction, obligation, violation
    entity_id = Column(String(64))
    
    # Event details
    event_data = Column(JSON)
    user_id = Column(String(64))
    session_id = Column(String(64))
    
    # Compliance metadata
    data_hash = Column(String(64))  # Hash of sensitive data
    compliance_flags = Column(JSON)
    
    # Metadata
    timestamp = Column(DateTime, server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
        Index('idx_audit_event', 'event_type'),
    )


# Database configuration
def get_database_url():
    """Get database URL from environment or default"""
    return os.getenv(
        'DATABASE_URL',
        'postgresql://finlex_user:finlex_pass@localhost:5432/finlex_db'
    )


def create_engine_instance():
    """Create SQLAlchemy engine with optimized settings"""
    return create_engine(
        get_database_url(),
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
    )


def get_session_maker():
    """Get session maker for database operations"""
    engine = create_engine_instance()
    return sessionmaker(bind=engine)


def init_database():
    """Initialize database tables"""
    engine = create_engine_instance()
    Base.metadata.create_all(engine)
    print("Database initialized successfully")


if __name__ == "__main__":
    init_database()