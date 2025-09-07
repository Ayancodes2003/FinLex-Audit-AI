-- FinLex Audit AI Database Initialization Script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database schema
CREATE SCHEMA IF NOT EXISTS finlex;

-- Set default schema
SET search_path = finlex, public;

-- Create tables (these will be managed by SQLAlchemy migrations, but we include them for reference)
CREATE TABLE IF NOT EXISTS policy_documents (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    document_hash VARCHAR(64) UNIQUE NOT NULL,
    source VARCHAR(100),
    jurisdiction VARCHAR(10) DEFAULT 'US',
    effective_date TIMESTAMP,
    expiry_date TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS obligations (
    id VARCHAR(255) PRIMARY KEY,
    policy_document_id VARCHAR(255) REFERENCES policy_documents(id),
    actor VARCHAR(500),
    action TEXT,
    type VARCHAR(50),
    condition TEXT,
    jurisdiction VARCHAR(10),
    temporal_scope VARCHAR(200),
    confidence FLOAT,
    extraction_model VARCHAR(100),
    source_clause TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
    id VARCHAR(255) PRIMARY KEY,
    step INTEGER,
    type VARCHAR(50),
    amount DECIMAL(15,2),
    name_orig VARCHAR(255),
    oldbalance_orig DECIMAL(15,2),
    newbalance_orig DECIMAL(15,2),
    name_dest VARCHAR(255),
    oldbalance_dest DECIMAL(15,2),
    newbalance_dest DECIMAL(15,2),
    is_fraud BOOLEAN DEFAULT FALSE,
    is_flagged BOOLEAN DEFAULT FALSE,
    currency VARCHAR(3) DEFAULT 'USD',
    amount_usd DECIMAL(15,2),
    timestamp TIMESTAMP,
    country VARCHAR(3),
    name_orig_hash VARCHAR(64),
    name_dest_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS violations (
    id VARCHAR(255) PRIMARY KEY,
    transaction_id VARCHAR(255) REFERENCES transactions(id),
    obligation_id VARCHAR(255) REFERENCES obligations(id),
    violation_type VARCHAR(100),
    confidence FLOAT,
    risk_level VARCHAR(20),
    reasoning TEXT,
    recommended_action VARCHAR(100),
    review_status VARCHAR(20) DEFAULT 'pending',
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP,
    resolution TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id VARCHAR(255) PRIMARY KEY,
    event_type VARCHAR(100),
    entity_type VARCHAR(50),
    entity_id VARCHAR(255),
    event_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    ip_address INET
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_policy_documents_hash ON policy_documents(document_hash);
CREATE INDEX IF NOT EXISTS idx_obligations_policy_id ON obligations(policy_document_id);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(type);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_transaction_id ON violations(transaction_id);
CREATE INDEX IF NOT EXISTS idx_violations_review_status ON violations(review_status);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- Insert sample data for testing
INSERT INTO policy_documents (id, title, content, document_hash, source, jurisdiction) 
VALUES (
    'pol_sample_001', 
    'Sample AML Policy', 
    'Financial institutions must report currency transactions exceeding $10,000 to FinCEN within 15 days. Banks are prohibited from processing transactions involving sanctioned entities without proper authorization.',
    'sample_hash_001',
    'system_init',
    'US'
) ON CONFLICT (document_hash) DO NOTHING;

INSERT INTO obligations (id, policy_document_id, actor, action, type, condition, jurisdiction, confidence, extraction_model, source_clause)
VALUES (
    'obl_sample_001',
    'pol_sample_001',
    'financial institutions',
    'report currency transactions exceeding $10,000 to FinCEN',
    'requirement',
    'transaction amount > $10,000',
    'US',
    0.95,
    'gemini-1.5-flash',
    'Financial institutions must report currency transactions exceeding $10,000 to FinCEN within 15 days.'
) ON CONFLICT (id) DO NOTHING;

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_policy_documents_updated_at 
    BEFORE UPDATE ON policy_documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions to finlex_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA finlex TO finlex_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA finlex TO finlex_user;
GRANT USAGE ON SCHEMA finlex TO finlex_user;