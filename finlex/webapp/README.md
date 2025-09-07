# FinLex Audit AI - Web Application

🏦 **Professional Financial Compliance Analysis Platform**

A modern web application for AI-powered financial compliance analysis, built with Flask backend, HTML/CSS/JS frontend, and Docker deployment.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Git (for cloning the repository)
- Gemini API key from Google AI Studio

### 1. Clone and Setup

```bash
# Navigate to the webapp directory
cd finlex/webapp

# Copy environment template
cp .env.example .env

# Edit .env file and add your Gemini API key
# GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Start the Application

**Windows:**
```bash
start.bat
```

**Linux/MacOS:**
```bash
chmod +x start.sh
./start.sh
```

**Manual Docker:**
```bash
docker-compose up -d
```

### 3. Access the Application

- 🌐 **Web Application**: http://localhost:5000
- 📊 **Nginx Proxy**: http://localhost:80  
- 🗄️ **Database**: localhost:5432 (PostgreSQL)
- 🔴 **Redis**: localhost:6379

## 🏗️ Architecture

### Technology Stack

- **Backend**: Flask 3.0 with SQLAlchemy ORM
- **Frontend**: Modern HTML5, CSS3, JavaScript (Vanilla JS)
- **Database**: PostgreSQL 15 with Redis caching
- **AI/ML**: Google Gemini-Flash LLM integration
- **Deployment**: Docker Compose with Nginx reverse proxy

### Services

1. **Web Application** (`webapp`)
   - Flask REST API backend
   - Serves HTML/CSS/JS frontend
   - Handles file uploads and processing
   - AI-powered compliance analysis

2. **PostgreSQL Database** (`postgres`)
   - Stores transactions, policies, obligations, violations
   - Automatic schema initialization
   - Sample data for testing

3. **Redis Cache** (`redis`)
   - Session storage and caching
   - Improves performance for repeated operations

4. **Nginx Proxy** (`nginx`)
   - Reverse proxy and load balancer
   - Rate limiting and security headers
   - SSL termination (production ready)

## 📋 Features

### Dashboard
- 📊 Real-time compliance metrics
- 🚨 Recent violations overview
- ⚡ System health monitoring
- 📈 Transaction statistics

### Transaction Management
- 📤 CSV file upload (Paysim1 format)
- 💰 Transaction processing and normalization
- 🔍 Automatic compliance scanning
- 📊 Risk assessment and flagging

### Policy Management
- 📋 Policy document upload (TXT/MD)
- 🤖 AI-powered obligation extraction
- 📝 Structured obligation database
- 🎯 Jurisdiction-specific rules

### Compliance Analysis
- 🔍 Real-time transaction scanning
- 🚨 Violation detection and alerting
- 📊 Risk level assessment
- 🎯 Configurable thresholds

### Violation Review
- 📋 Violation workflow management
- 👤 Review assignment and tracking
- 📝 Resolution documentation
- 📊 Audit trail and reporting

## 🗄️ Database Schema

### Core Tables

- **`policy_documents`**: Policy files and metadata
- **`obligations`**: Extracted compliance requirements
- **`transactions`**: Financial transaction records
- **`violations`**: Detected compliance violations
- **`audit_logs`**: System activity audit trail

### Sample Data

The system includes sample policies and obligations for testing:
- AML (Anti-Money Laundering) policies
- Transaction reporting requirements
- Sanctioned entity checks

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Database
DATABASE_URL=postgresql://finlex_user:finlex_pass@postgres:5432/finlex_db
POSTGRES_DB=finlex_db
POSTGRES_USER=finlex_user
POSTGRES_PASSWORD=finlex_pass

# Redis
REDIS_URL=redis://redis:6379/0

# AI Integration
GEMINI_API_KEY=your_gemini_api_key_here

# Flask
SECRET_KEY=your-super-secret-key
FLASK_ENV=production
```

### Compliance Rules

Default thresholds by jurisdiction:

**US Regulations:**
- Large transactions: $100,000
- Cash reporting: $10,000
- Suspicious frequency: 5+ transactions
- High velocity: $50,000

**EU Regulations:**
- Large transactions: €85,000
- Cash reporting: €8,500
- Suspicious frequency: 5+ transactions
- High velocity: €42,500

## 🧪 API Endpoints

### Health & Status
- `GET /api/health` - System health check
- `GET /api/dashboard/stats` - Dashboard statistics

### Transactions
- `POST /api/transactions/upload` - Upload transaction CSV
- `POST /api/transactions/scan` - Scan for violations
- `GET /api/transactions` - List transactions

### Policies
- `POST /api/policies/upload` - Upload policy document
- `GET /api/policies/{id}/obligations` - Get extracted obligations

### Compliance
- `POST /api/compliance/scan` - Run compliance analysis
- `GET /api/violations` - List violations
- `PUT /api/violations/{id}/review` - Update violation review

## 🚦 Usage Workflow

### 1. Upload Policy Documents
1. Navigate to **Policies** tab
2. Enter policy title and jurisdiction
3. Upload TXT or MD file
4. AI extracts structured obligations
5. Review and confirm obligations

### 2. Upload Transaction Data
1. Navigate to **Transactions** tab
2. Upload CSV file (Paysim1 format)
3. System processes and normalizes data
4. Automatic PII hashing for security

### 3. Run Compliance Analysis
1. Go to **Compliance** tab
2. Configure scan parameters
3. Run analysis against uploaded data
4. Review detected violations

### 4. Review Violations
1. Navigate to **Violations** tab
2. Review flagged transactions
3. Assign risk levels and actions
4. Document resolutions
5. Update compliance status

## 🛠️ Development

### Local Development Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up database
export DATABASE_URL="postgresql://user:pass@localhost:5432/finlex_dev"
flask db init
flask db migrate -m "Initial migration"
flask db upgrade

# Run development server
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

### Docker Development

```bash
# Build and run in development mode
docker-compose -f docker-compose.dev.yml up --build

# View logs
docker-compose logs -f webapp

# Execute database migrations
docker-compose exec webapp flask db upgrade
```

## 📊 Monitoring

### Service Health Checks

```bash
# Check all services
docker-compose ps

# Check application health
curl http://localhost:5000/api/health

# Check database connection
docker-compose exec postgres pg_isready -U finlex_user
```

### Logs

```bash
# Application logs
docker-compose logs -f webapp

# Database logs  
docker-compose logs -f postgres

# All services
docker-compose logs -f
```

## 🔒 Security Features

### Data Protection
- 🔐 PII hashing for sensitive transaction data
- 🛡️ SQL injection prevention with ORM
- 🔒 CORS protection for API endpoints
- 📋 Comprehensive audit logging

### Infrastructure Security  
- 🚫 Rate limiting on API endpoints
- 🛡️ Security headers via Nginx
- 🔐 Environment variable configuration
- 📊 Health check monitoring

## 🐛 Troubleshooting

### Common Issues

**Docker Services Won't Start:**
```bash
# Check Docker is running
docker --version
docker-compose --version

# Check ports are available
netstat -an | findstr ":5000"
netstat -an | findstr ":5432"
```

**Database Connection Issues:**
```bash
# Check PostgreSQL service
docker-compose logs postgres

# Test database connection
docker-compose exec postgres psql -U finlex_user -d finlex_db -c "SELECT 1;"
```

**Gemini API Errors:**
- Verify API key in `.env` file
- Check API quota and billing
- Review network connectivity

### Reset Everything

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Rebuild and restart
docker-compose up --build -d
```

## 📈 Performance Optimization

### Production Recommendations

1. **Database Optimization**
   - Configure PostgreSQL connection pooling
   - Add database indexes for large datasets
   - Regular VACUUM and ANALYZE operations

2. **Application Scaling**
   - Use Gunicorn with multiple workers
   - Implement Redis caching for frequent queries
   - Add application-level rate limiting

3. **Infrastructure**
   - SSL/TLS certificates for HTTPS
   - CDN for static assets
   - Load balancer for multiple instances

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for LLM capabilities
- PaySim dataset for transaction simulation
- ObliQA dataset for obligation extraction examples
- Flask and SQLAlchemy communities

---

**FinLex Audit AI** - Bringing AI-powered compliance analysis to financial institutions worldwide 🌍