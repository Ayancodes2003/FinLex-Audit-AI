# FinLex Audit AI

🏦 **AI-Powered Financial Compliance Analysis System**

A production-grade compliance audit system that ingests policies and transactions, extracts obligations, detects violations using rules and semantic matching, and generates explainable reports.

## 🚀 Features

- **Policy Processing**: Extract structured obligations from policy documents using Gemini-Flash with few-shot learning
- **Transaction Analysis**: Ingest and normalize Paysim1-like transaction data with PII protection
- **Compliance Detection**: Deterministic rules + semantic matching with FAISS vector search
- **Explainable Reports**: RAG-powered violation explanations with regulatory references
- **Interactive Dashboard**: Streamlit UI for policy upload, transaction scanning, and violation review
- **Production Ready**: Docker deployment, comprehensive testing, audit logging

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◄──►│   Main API       │◄──►│   Database      │
│   (Port 8501)   │    │   (Port 8000)    │    │   PostgreSQL    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ Ingest Service  │ │Extractor Service│ │ Matcher Service │
    │  (Port 8001)    │ │  (Port 8002)    │ │  (Port 8003)    │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                      ┌─────────────────┐
                      │RAG Generator    │
                      │  (Port 8004)    │
                      └─────────────────┘
                                │
                                ▼
                      ┌─────────────────┐
                      │ Gemini-Flash    │
                      │ Vector DB (FAISS)│
                      └─────────────────┘
```

## 🛠️ Technology Stack

- **Backend**: FastAPI microservices, Python 3.11+
- **LLM**: Gemini-Flash for obligation extraction and explanations
- **Database**: PostgreSQL for structured data, FAISS for embeddings
- **Frontend**: Streamlit dashboard
- **Infrastructure**: Docker Compose, pytest testing
- **Security**: PII hashing, audit logging, compliance tracking

## 📦 Repository Structure

```
finlex/
├── services/
│   ├── gemini_client.py      # Centralized Gemini-Flash client
│   ├── database.py           # SQLAlchemy models and DB setup
│   ├── ingest/main.py        # Transaction ingestion service
│   ├── extractor/main.py     # Policy obligation extractor
│   ├── matcher/main.py       # Compliance rule matcher
│   ├── raggenerator/main.py  # RAG violation explanations
│   └── api/main.py           # Main orchestration API
├── ui/app.py                 # Streamlit dashboard
├── infra/
│   ├── docker-compose.yml    # Full deployment setup
│   ├── .env.example          # Environment configuration
│   └── Dockerfile.*          # Service containers
├── tests/test_rules.py       # Comprehensive test suite
├── requirements.txt          # Python dependencies
├── requirements_ui.txt       # Streamlit dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Gemini API key (set `GEMINI_API_KEY` environment variable)

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/FinLex-Audit-AI.git
cd FinLex-Audit-AI/finlex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_ui.txt
```

### 2. Environment Configuration

```bash
# Copy and configure environment
cp infra/.env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Start with Docker (Recommended)

```bash
# Start all services
cd infra
docker-compose up -d

# Check service health
docker-compose ps
```

### 4. Start Development Mode

```bash
# Terminal 1: Start API services
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Streamlit UI
streamlit run ui/app.py --server.port 8501
```

### 5. Access the Dashboard

Open http://localhost:8501 in your browser to access the FinLex dashboard.

## 📋 Usage Guide

### Upload Policy Documents

1. Navigate to **Policy Management** in the dashboard
2. Upload `.txt` or `.md` policy files
3. Configure jurisdiction and few-shot learning options
4. Review extracted obligations with confidence scores

### Ingest Transaction Data

1. Go to **Upload Data** section
2. Upload CSV files in Paysim1 format:
   ```csv
   step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedMerchant
   1,TRANSFER,150000,C1234567890,200000,50000,M9876543210,0,150000,0,0
   ```
3. Monitor processing status and any validation errors

### Run Compliance Scans

1. Access **Compliance Scan** feature
2. Select transactions to analyze (recent, range, or custom list)
3. Configure jurisdiction and confidence thresholds
4. Review detected violations with:
   - Risk levels (Low/Medium/High/Critical)
   - Confidence scores
   - Detailed explanations
   - Regulatory references
   - Recommended actions

### Review Violations

1. Use **Violation Review** to approve/reject findings
2. Add review notes and track audit trail
3. Export reports for regulatory filing

## 🔧 API Endpoints

### Main API (Port 8000)
- `POST /scan` - Run compliance scan
- `POST /transactions/upload` - Upload transaction CSV
- `POST /policies/upload` - Upload policy document
- `GET /dashboard/stats` - Dashboard statistics
- `GET /violations/recent` - Recent violations
- `GET /health` - System health check

### Service APIs
- **Ingest** (8001): `POST /transactions`, `GET /transactions/{id}`
- **Extractor** (8002): `POST /extract`, `POST /extract/file`
- **Matcher** (8003): `POST /match`, `GET /rules/thresholds/{jurisdiction}`
- **RAG Generator** (8004): `POST /generate`, `PUT /violations/{id}/review`

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_rules.py::TestThresholdRules -v
pytest tests/test_rules.py::TestGeminiClient -v
pytest tests/test_rules.py::TestIntegration -v

# Run with coverage
pytest tests/ --cov=services --cov-report=html
```

## 🔒 Security & Compliance

- **PII Protection**: All personal identifiers hashed before LLM processing
- **Audit Logging**: Complete audit trail of all LLM calls and decisions
- **Data Governance**: Structured data storage with retention policies
- **Access Control**: Role-based violation review and approval
- **Encryption**: All data encrypted in transit and at rest

## 🚀 Deployment

### Production Deployment

```bash
# Set production environment
export ENVIRONMENT=production
export GEMINI_API_KEY=your_production_key
export DATABASE_URL=your_production_db_url

# Deploy with Docker
docker-compose -f infra/docker-compose.yml up -d

# Or deploy to cloud (example: Google Cloud Run)
gcloud run deploy finlex-api --source=services/
```

### Scaling Considerations

- **Horizontal Scaling**: Each service can be scaled independently
- **Database Optimization**: Indexed queries for high-volume transaction processing
- **Caching**: Redis integration for frequently accessed data
- **Rate Limiting**: Built-in request throttling for LLM API calls

## 📊 Monitoring

- **Health Checks**: `/health` endpoints for all services
- **Metrics**: Prometheus-compatible metrics collection
- **Logging**: Structured logging with compliance audit trails
- **Alerts**: Configurable alerts for violation detection and system health

## 🔄 Data Pipeline

1. **Ingestion**: Normalize transactions (UTC timestamps, USD currency, PII hashing)
2. **Policy Processing**: Split documents → Extract obligations → Store in vector DB
3. **Detection**: Apply threshold rules → Semantic matching → Collect evidence
4. **Explanation**: RAG-powered analysis → Generate human-readable reports
5. **Review**: Human approval/rejection → Audit trail → Regulatory filing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `pytest tests/ -v`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: See `/docs` directory for detailed guides
- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions for questions and best practices

## 🏗️ Roadmap

- [ ] Neo4j integration for temporal/jurisdiction knowledge graphs
- [ ] Real-time streaming transaction processing
- [ ] Multi-language policy support
- [ ] Advanced ML models for anomaly detection
- [ ] Regulatory reporting automation
- [ ] Integration with major banking APIs

---

**Built with ❤️ for financial compliance teams worldwide**
