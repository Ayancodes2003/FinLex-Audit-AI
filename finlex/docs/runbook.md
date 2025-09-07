# FinLex Audit AI - Operations Runbook

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Pre-deployment Setup](#pre-deployment-setup)
3. [Data Preparation](#data-preparation)
4. [Deployment Procedures](#deployment-procedures)
5. [Operation Modes](#operation-modes)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Backup and Recovery](#backup-and-recovery)

## 🏗️ System Overview

FinLex Audit AI is a microservices-based compliance analysis platform consisting of:

- **Ingest Service** (Port 8001): Transaction data ingestion and normalization
- **Extractor Service** (Port 8002): Policy obligation extraction using Gemini-Flash
- **Matcher Service** (Port 8003): Compliance rule matching (deterministic + semantic)
- **RAG Generator** (Port 8004): Explainable violation report generation
- **Main API** (Port 8000): Orchestration and unified interface
- **Streamlit UI** (Port 8501): Interactive dashboard
- **PostgreSQL**: Structured data storage
- **FAISS Vector DB**: Semantic embeddings storage

## 🚀 Pre-deployment Setup

### Environment Configuration

1. **Create environment file:**
```bash
cp finlex/infra/.env.example finlex/.env
```

2. **Configure required variables:**
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://finlex_user:finlex_pass@localhost:5432/finlex_db

# Optional
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

### Database Initialization

```bash
# Start PostgreSQL
docker run -d --name finlex-postgres \
  -e POSTGRES_DB=finlex_db \
  -e POSTGRES_USER=finlex_user \
  -e POSTGRES_PASSWORD=finlex_pass \
  -p 5432:5432 postgres:15-alpine

# Initialize database schema
cd finlex
python -c "from services.database import init_database; init_database()"
```

### Gemini API Setup

1. **Obtain API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create new API key
   - Set in environment: `export GEMINI_API_KEY=your_key_here`

2. **Test API Connection:**
```bash
python -c "
from services.gemini_client import get_gemini_client
client = get_gemini_client()
print('Gemini client initialized successfully')
"
```

## 📊 Data Preparation

### Preprocessing Paysim1 Dataset

```bash
# Download Paysim1 dataset
wget https://www.kaggle.com/datasets/ealaxi/paysim1/download -O paysim1.zip
unzip paysim1.zip

# Preprocess for FinLex format
python scripts/preprocess_paysim1.py \
  --input PS_20174392719_1491204439457_log.csv \
  --output processed_transactions.csv \
  --sample-size 100000 \
  --normalize-currency USD
```

**Expected CSV format:**
```csv
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedMerchant
1,TRANSFER,150000,C1234567890,200000,50000,M9876543210,0,150000,0,0
```

### Loading ObliQa Dataset

```bash
# Download ObliQa obligation QA dataset
git clone https://github.com/trusthlt/ObliQA.git data/obliqa

# Process for few-shot examples
python scripts/process_obliqa.py \
  --input data/obliqa/train.json \
  --output services/extractor/obliqa_examples.json \
  --max-examples 100
```

### Loading C3PA Policy Dataset

```bash
# Download C3PA compliance clauses
git clone https://github.com/maastrichtlawtech/c3pa.git data/c3pa

# Process policy documents
python scripts/process_c3pa.py \
  --input data/c3pa/policies/ \
  --output data/processed_policies/ \
  --jurisdiction US \
  --format finlex
```

### Building FAISS Index

```bash
# Create vector embeddings for obligations
python scripts/build_faiss_index.py \
  --policy-dir data/processed_policies/ \
  --output-index data/obligations.index \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

## 🐳 Deployment Procedures

### Development Deployment

```bash
# Method 1: Docker Compose (Recommended)
cd finlex/infra
docker-compose up -d

# Method 2: Manual Services
# Terminal 1: Start PostgreSQL
docker run -d --name finlex-postgres -p 5432:5432 \
  -e POSTGRES_DB=finlex_db -e POSTGRES_USER=finlex_user \
  -e POSTGRES_PASSWORD=finlex_pass postgres:15-alpine

# Terminal 2: Start services
cd finlex
python -m uvicorn services.api.main:app --port 8000 --reload &
python -m uvicorn services.ingest.main:app --port 8001 --reload &
python -m uvicorn services.extractor.main:app --port 8002 --reload &
python -m uvicorn services.matcher.main:app --port 8003 --reload &
python -m uvicorn services.raggenerator.main:app --port 8004 --reload &

# Terminal 3: Start UI
streamlit run ui/app.py --server.port 8501
```

### Production Deployment

```bash
# Build and deploy with Docker
cd finlex/infra
export ENVIRONMENT=production
export GEMINI_API_KEY=your_production_key
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

### Cloud Deployment (Google Cloud)

```bash
# Deploy API service
gcloud run deploy finlex-api \
  --source=services/ \
  --platform=managed \
  --region=us-central1 \
  --set-env-vars="GEMINI_API_KEY=$GEMINI_API_KEY"

# Deploy UI
gcloud run deploy finlex-ui \
  --source=ui/ \
  --platform=managed \
  --region=us-central1 \
  --set-env-vars="API_BASE_URL=https://finlex-api-xyz.run.app"
```

## ⚙️ Operation Modes

### Shadow Mode

Run compliance analysis without taking enforcement actions:

```bash
# Enable shadow mode
export COMPLIANCE_MODE=shadow

# Start services with shadow mode
docker-compose up -d

# All violations will be detected and logged but no enforcement actions taken
```

### Active Mode

Full compliance enforcement with real-time actions:

```bash
# Enable active mode
export COMPLIANCE_MODE=active
export ENABLE_REAL_TIME_BLOCKING=true

# Start services with active enforcement
docker-compose up -d
```

### Batch Processing Mode

Process large transaction volumes offline:

```bash
# Run batch compliance scan
python scripts/batch_compliance_scan.py \
  --input large_transaction_file.csv \
  --output compliance_report.json \
  --batch-size 1000 \
  --jurisdiction US
```

## 📊 Monitoring and Maintenance

### Health Monitoring

```bash
# Check all service health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "services": {
    "ingest": "healthy",
    "extractor": "healthy", 
    "matcher": "healthy",
    "raggenerator": "healthy"
  }
}
```

### Performance Monitoring

```bash
# Monitor API response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/dashboard/stats

# Monitor database connections
docker exec finlex-postgres psql -U finlex_user -d finlex_db -c "\
SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"
```

### Log Analysis

```bash
# View service logs
docker-compose logs -f api-service
docker-compose logs -f extractor-service --tail=100

# Search for compliance violations
docker-compose logs | grep "violation_detected.*true"

# Monitor LLM call audit trail
docker-compose logs | grep "LLM_CALL" | jq .
```

### Resource Usage

```bash
# Monitor container resource usage
docker stats finlex-api finlex-postgres finlex-ui

# Monitor disk usage
du -sh data/vector_indexes/
du -sh postgres_data/
```

## 🔧 Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check port conflicts
netstat -tulpn | grep -E ":(8000|8001|8002|8003|8004|8501|5432)"

# Check environment variables
docker-compose config

# View detailed error logs
docker-compose logs service-name
```

#### Database Connection Issues

```bash
# Test database connectivity
docker exec finlex-postgres pg_isready -U finlex_user -d finlex_db

# Reset database
docker-compose down -v
docker-compose up -d postgres
# Wait 30 seconds
python -c "from services.database import init_database; init_database()"
```

#### Gemini API Issues

```bash
# Test API key
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  https://generativelanguage.googleapis.com/v1/models

# Check rate limits
grep "rate_limit" logs/gemini_client.log
```

#### FAISS Index Issues

```bash
# Rebuild FAISS index
python scripts/build_faiss_index.py \
  --rebuild \
  --policy-dir data/processed_policies/ \
  --output-index data/obligations.index
```

### Performance Issues

#### Slow Compliance Scans

```bash
# Check transaction volume
psql $DATABASE_URL -c "SELECT COUNT(*) FROM transactions;"

# Optimize batch sizes
export COMPLIANCE_BATCH_SIZE=100  # Reduce from default 500

# Enable caching
export ENABLE_OBLIGATION_CACHE=true
```

#### High Memory Usage

```bash
# Monitor service memory
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Reduce FAISS index size
export FAISS_INDEX_DIMENSION=128  # Reduce from 512
python scripts/rebuild_faiss_index.py --dimension 128
```

## 💾 Backup and Recovery

### Database Backup

```bash
# Create database backup
docker exec finlex-postgres pg_dump -U finlex_user finlex_db > backup_$(date +%Y%m%d).sql

# Automated daily backups
crontab -e
# Add: 0 2 * * * /path/to/backup_script.sh
```

### Vector Index Backup

```bash
# Backup FAISS indices
tar -czf vector_backup_$(date +%Y%m%d).tar.gz data/obligations.index

# Upload to cloud storage
gsutil cp vector_backup_*.tar.gz gs://finlex-backups/
```

### Disaster Recovery

```bash
# Restore from backup
docker exec -i finlex-postgres psql -U finlex_user finlex_db < backup_20241201.sql

# Restore vector indices
tar -xzf vector_backup_20241201.tar.gz -C data/

# Restart services
docker-compose restart
```

### Configuration Backup

```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
  finlex/.env \
  finlex/infra/docker-compose.yml \
  finlex/infra/*.env

# Store securely
gpg --symmetric --cipher-algo AES256 config_backup_*.tar.gz
```

## 🔒 Security Procedures

### PII Data Handling

```bash
# Verify PII hashing is working
python -c "
from services.ingest.main import TransactionProcessor
proc = TransactionProcessor()
print('Original:', 'John Doe')
print('Hashed:', proc.hash_pii('John Doe'))
"
```

### Audit Log Review

```bash
# Review compliance audit trail
psql $DATABASE_URL -c "
SELECT event_type, COUNT(*) 
FROM audit_logs 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY event_type;
"

# Export audit logs for compliance
psql $DATABASE_URL -c "\copy (
  SELECT * FROM audit_logs 
  WHERE timestamp >= '2024-01-01'
) TO 'audit_export_$(date +%Y%m%d).csv' CSV HEADER;"
```

### Certificate Management

```bash
# Renew SSL certificates (if using HTTPS)
certbot renew --nginx

# Update Docker secrets
docker secret create finlex_ssl_cert_v2 ssl_certificate.pem
docker service update --secret-rm finlex_ssl_cert --secret-add finlex_ssl_cert_v2 finlex_api
```

## 📈 Scaling Operations

### Horizontal Scaling

```bash
# Scale individual services
docker-compose up -d --scale matcher-service=3 --scale extractor-service=2

# Load balancer configuration (nginx)
upstream finlex_api {
    server finlex-api-1:8000;
    server finlex-api-2:8000;
    server finlex-api-3:8000;
}
```

### Database Scaling

```bash
# Enable PostgreSQL replication
# Primary database configuration
echo "wal_level = replica" >> postgresql.conf
echo "max_wal_senders = 3" >> postgresql.conf

# Read replica setup
docker run -d --name finlex-postgres-replica \
  -e PGUSER=replicator \
  -e POSTGRES_MASTER_SERVICE=finlex-postgres \
  postgres:15-alpine
```

## 🚨 Alert Configuration

### Prometheus Monitoring

```yaml
# prometheus.yml
rule_files:
  - "finlex_alerts.yml"

# finlex_alerts.yml
groups:
- name: finlex.rules
  rules:
  - alert: HighViolationRate
    expr: violation_rate > 0.1
    for: 5m
    annotations:
      summary: "High compliance violation rate detected"
  
  - alert: ServiceDown
    expr: up{job="finlex-api"} == 0
    for: 1m
    annotations:
      summary: "FinLex service is down"
```

### Email Notifications

```bash
# Configure email alerts
export SMTP_HOST=smtp.company.com
export SMTP_PORT=587
export ALERT_EMAIL=compliance-team@company.com

# Test email configuration
python scripts/test_email_alerts.py
```

---

For additional support, see the main [README.md](../README.md) or contact the development team.