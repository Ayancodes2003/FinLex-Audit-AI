"""
Database models for FinLex Audit AI Web Application

SQLAlchemy models that mirror the main services database structure.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()

class PolicyDocument(db.Model):
    __tablename__ = 'policy_documents'
    
    id = db.Column(db.String(255), primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    document_hash = db.Column(db.String(64), unique=True, nullable=False)
    source = db.Column(db.String(100))
    jurisdiction = db.Column(db.String(10), default='US')
    effective_date = db.Column(db.DateTime)
    expiry_date = db.Column(db.DateTime)
    is_processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    obligations = db.relationship('Obligation', backref='policy_document', lazy=True)


class Obligation(db.Model):
    __tablename__ = 'obligations'
    
    id = db.Column(db.String(255), primary_key=True)
    policy_document_id = db.Column(db.String(255), db.ForeignKey('policy_documents.id'), nullable=True)
    actor = db.Column(db.String(500))
    action = db.Column(db.Text)
    type = db.Column(db.String(50))
    condition = db.Column(db.Text)
    jurisdiction = db.Column(db.String(10))
    temporal_scope = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    extraction_model = db.Column(db.String(100))
    source_clause = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    violations = db.relationship('Violation', backref='obligation', lazy=True)


class Transaction(db.Model):
    __tablename__ = 'transactions'
    
    id = db.Column(db.String(255), primary_key=True)
    step = db.Column(db.Integer)
    type = db.Column(db.String(50))
    amount = db.Column(db.Numeric(15,2))
    name_orig = db.Column(db.String(255))
    oldbalance_orig = db.Column(db.Numeric(15,2))
    newbalance_orig = db.Column(db.Numeric(15,2))
    name_dest = db.Column(db.String(255))
    oldbalance_dest = db.Column(db.Numeric(15,2))
    newbalance_dest = db.Column(db.Numeric(15,2))
    is_fraud = db.Column(db.Boolean, default=False)
    is_flagged = db.Column(db.Boolean, default=False)
    currency = db.Column(db.String(3), default='USD')
    amount_usd = db.Column(db.Numeric(15,2))
    timestamp = db.Column(db.DateTime)
    country = db.Column(db.String(3))
    name_orig_hash = db.Column(db.String(64))
    name_dest_hash = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    violations = db.relationship('Violation', backref='transaction', lazy=True)


class Violation(db.Model):
    __tablename__ = 'violations'
    
    id = db.Column(db.String(255), primary_key=True)
    transaction_id = db.Column(db.String(255), db.ForeignKey('transactions.id'), nullable=True)
    obligation_id = db.Column(db.String(255), db.ForeignKey('obligations.id'), nullable=True)
    violation_type = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    reasoning = db.Column(db.Text)
    recommended_action = db.Column(db.String(100))
    review_status = db.Column(db.String(20), default='pending')
    reviewed_by = db.Column(db.String(255))
    reviewed_at = db.Column(db.DateTime)
    resolution = db.Column(db.Text)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)


class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type = db.Column(db.String(100))
    entity_type = db.Column(db.String(50))
    entity_id = db.Column(db.String(255))
    event_data = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.String(255))
    ip_address = db.Column(db.String(45))