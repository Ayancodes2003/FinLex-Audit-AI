"""
FinLex Audit AI - Flask Backend Application (Clean Version)

Professional Flask REST API for financial compliance analysis.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_migrate import Migrate
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
import json
import hashlib
import uuid
from typing import Dict, Any, List

# Local imports  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import db, Transaction, Obligation, Violation, PolicyDocument, AuditLog

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 
    'sqlite:///finlex_dev.db'  # Use SQLite for development
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)
CORS(app)

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv', 'md', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === ROUTES ===

@app.route('/')
def index():
    """Serve the main application"""
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/dashboard/stats')
def dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Count records from database
        transaction_count = db.session.query(Transaction).count()
        policy_count = db.session.query(PolicyDocument).count()
        obligation_count = db.session.query(Obligation).count()
        violation_count = db.session.query(Violation).count()
        
        # Recent violations (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_violations = db.session.query(Violation).filter(
            Violation.detected_at >= week_ago
        ).count()
        
        # Pending reviews
        pending_reviews = db.session.query(Violation).filter(
            Violation.review_status == 'pending'
        ).count()
        
        return jsonify({
            'total_transactions': transaction_count,
            'total_policies': policy_count,
            'total_obligations': obligation_count,
            'total_violations': violation_count,
            'recent_violations': recent_violations,
            'pending_reviews': pending_reviews,
            'system_status': 'operational'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/routes')
def list_routes():
    """Debug endpoint to list all routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': rule.rule
        })
    return jsonify(routes)

@app.route('/api/demo/generate-sample-data', methods=['POST'])
def generate_sample_data():
    """Generate sample data for testing"""
    try:
        # Create sample policy document
        sample_policy = PolicyDocument(
            id='pol_sample_001',
            title='Sample AML Policy',
            content='Financial institutions must report currency transactions exceeding $10,000 to FinCEN within 15 days. Banks are prohibited from processing transactions involving sanctioned entities without proper authorization.',
            document_hash='sample_hash_001',
            source='demo_generation',
            jurisdiction='US',
            is_processed=True
        )
        
        # Check if already exists
        existing_policy = PolicyDocument.query.filter_by(id='pol_sample_001').first()
        if not existing_policy:
            db.session.add(sample_policy)
        
        # Create sample obligation
        sample_obligation = Obligation(
            id='obl_sample_001',
            policy_document_id='pol_sample_001',
            actor='financial institutions',
            action='report currency transactions exceeding $10,000 to FinCEN',
            type='requirement',
            condition='transaction amount > $10,000',
            jurisdiction='US',
            confidence=0.95,
            extraction_model='gemini-1.5-flash',
            source_clause='Financial institutions must report currency transactions exceeding $10,000 to FinCEN within 15 days.'
        )
        
        existing_obligation = Obligation.query.filter_by(id='obl_sample_001').first()
        if not existing_obligation:
            db.session.add(sample_obligation)
        
        # Create sample transactions
        sample_transactions = [
            Transaction(
                id='txn_001',
                step=1,
                type='TRANSFER',
                amount=15000.00,
                name_orig='Alice Corp',
                oldbalance_orig=50000.00,
                newbalance_orig=35000.00,
                name_dest='Bob Industries',
                oldbalance_dest=25000.00,
                newbalance_dest=40000.00,
                amount_usd=15000.00,
                timestamp=datetime.utcnow(),
                country='US',
                name_orig_hash=hashlib.sha256('Alice Corp'.encode()).hexdigest(),
                name_dest_hash=hashlib.sha256('Bob Industries'.encode()).hexdigest()
            ),
            Transaction(
                id='txn_002',
                step=2,
                type='CASH_OUT',
                amount=12000.00,
                name_orig='Charlie Ltd',
                oldbalance_orig=30000.00,
                newbalance_orig=18000.00,
                name_dest='ATM_Network',
                oldbalance_dest=0.00,
                newbalance_dest=0.00,
                amount_usd=12000.00,
                timestamp=datetime.utcnow(),
                country='US',
                name_orig_hash=hashlib.sha256('Charlie Ltd'.encode()).hexdigest(),
                name_dest_hash=hashlib.sha256('ATM_Network'.encode()).hexdigest()
            )
        ]
        
        for txn in sample_transactions:
            existing_txn = Transaction.query.filter_by(id=txn.id).first()
            if not existing_txn:
                db.session.add(txn)
        
        # Create sample violations
        sample_violations = [
            Violation(
                id='vio_001',
                transaction_id='txn_001',
                obligation_id='obl_sample_001',
                violation_type='large_transaction',
                confidence=1.0,
                risk_level='medium',
                reasoning='Transaction amount $15,000.00 exceeds reporting threshold of $10,000.00',
                recommended_action='report',
                review_status='pending'
            ),
            Violation(
                id='vio_002',
                transaction_id='txn_002',
                obligation_id='obl_sample_001',
                violation_type='cash_reporting',
                confidence=1.0,
                risk_level='medium',
                reasoning='Cash transaction of $12,000.00 requires regulatory reporting',
                recommended_action='report',
                review_status='pending'
            )
        ]
        
        for vio in sample_violations:
            existing_vio = Violation.query.filter_by(id=vio.id).first()
            if not existing_vio:
                db.session.add(vio)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Sample data generated successfully',
            'data': {
                'policies': 1,
                'obligations': 1,
                'transactions': 2,
                'violations': 2
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to generate sample data: {str(e)}'}), 500

def create_tables():
    """Create database tables"""
    try:
        with app.app_context():
            db.create_all()
            print("✅ Database tables created successfully")
    except Exception as e:
        print(f"⚠️  Database setup warning: {e}")

if __name__ == '__main__':
    # Create tables on startup
    create_tables()
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )