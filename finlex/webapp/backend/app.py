"""
FinLex Audit AI - Flask Backend Application

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
import sys
import os
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


# === UTILITY CLASSES ===

class TransactionProcessor:
    """Process and normalize transaction data"""
    
    CURRENCY_RATES = {
        'USD': 1.0, 'EUR': 1.08, 'GBP': 1.25, 'JPY': 0.007, 'CAD': 0.74
    }
    
    @staticmethod
    def normalize_currency(amount: float, currency: str) -> float:
        rate = TransactionProcessor.CURRENCY_RATES.get(currency.upper(), 1.0)
        return amount * rate
    
    @staticmethod
    def hash_pii(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()
    
    @staticmethod
    def normalize_timestamp(timestamp_str: str) -> datetime:
        try:
            dt = pd.to_datetime(timestamp_str)
            if dt.tz is None:
                dt = dt.tz_localize(timezone.utc)
            return dt.tz_convert(timezone.utc).to_pydatetime()
        except:
            return datetime.now(timezone.utc)


class ComplianceRules:
    """Compliance rule engine"""
    
    THRESHOLDS = {
        'US': {
            'large_transaction': 100000.0,
            'cash_reporting': 10000.0,
            'suspicious_frequency': 5,
            'high_velocity': 50000.0
        },
        'EU': {
            'large_transaction': 85000.0,
            'cash_reporting': 8500.0,
            'suspicious_frequency': 5,
            'high_velocity': 42500.0
        }
    }
    
    @classmethod
    def check_violations(cls, transaction_data: Dict, jurisdiction: str = 'US') -> List[Dict]:
        thresholds = cls.THRESHOLDS.get(jurisdiction, cls.THRESHOLDS['US'])
        violations = []
        
        amount = transaction_data.get('amount_usd', 0)
        tx_type = transaction_data.get('type', '')
        
        # Large transaction check
        if amount > thresholds['large_transaction']:
            violations.append({
                'type': 'large_transaction',
                'confidence': 1.0,
                'risk_level': 'high' if amount > thresholds['large_transaction'] * 2 else 'medium',
                'reasoning': f'Transaction amount ${amount:,.2f} exceeds threshold of ${thresholds["large_transaction"]:,.2f}',
                'recommended_action': 'escalate' if amount > thresholds['large_transaction'] * 2 else 'review'
            })
        
        # Cash reporting check
        if tx_type in ['CASH_OUT', 'CASH_IN'] and amount > thresholds['cash_reporting']:
            violations.append({
                'type': 'cash_reporting',
                'confidence': 1.0,
                'risk_level': 'medium',
                'reasoning': f'Cash transaction of ${amount:,.2f} requires regulatory reporting',
                'recommended_action': 'report'
            })
        
        return violations


# === API ROUTES ===

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
        from datetime import timedelta
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

@app.route('/api/transactions/upload', methods=['POST'])
def upload_transactions():
    """Upload and process transaction CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read CSV data
        df = pd.read_csv(file)
        
        processor = TransactionProcessor()
        processed_transactions = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                # Normalize data
                amount_usd = processor.normalize_currency(
                    row.get('amount', 0), 
                    row.get('currency', 'USD')
                )
                
                # Hash PII
                name_orig_hash = processor.hash_pii(str(row.get('nameOrig', f'unknown_{idx}')))
                name_dest_hash = processor.hash_pii(str(row.get('nameDest', f'unknown_{idx}')))
                
                # Create transaction ID
                tx_id = hashlib.sha256(f"{name_orig_hash}_{name_dest_hash}_{amount_usd}_{idx}".encode()).hexdigest()
                
                # Create transaction record
                transaction = Transaction(
                    id=tx_id,
                    step=int(row.get('step', idx + 1)),
                    type=str(row.get('type', 'PAYMENT')).upper(),
                    amount_usd=amount_usd,
                    name_orig_hash=name_orig_hash,
                    old_balance_orig=float(row.get('oldbalanceOrg', 0)),
                    new_balance_orig=float(row.get('newbalanceOrig', 0)),
                    name_dest_hash=name_dest_hash,
                    old_balance_dest=float(row.get('oldbalanceDest', 0)),
                    new_balance_dest=float(row.get('newbalanceDest', 0)),
                    is_fraud=bool(row.get('isFraud', False)),
                    is_flagged_merchant=bool(row.get('isFlaggedMerchant', False)),
                    timestamp_utc=processor.normalize_timestamp(row.get('timestamp', datetime.now())),
                    amount_1d_total=amount_usd * 1.2,  # Mock derived features
                    amount_30d_total=amount_usd * 15.8,
                    transaction_count_1d=3,
                    transaction_count_30d=47
                )
                
                processed_transactions.append(transaction)
                
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        # Save to database
        try:
            for tx in processed_transactions:
                # Check for duplicates
                existing = db.session.query(Transaction).filter(Transaction.id == tx.id).first()
                if not existing:
                    db.session.add(tx)
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'processed_count': len(processed_transactions),
                'failed_count': len(errors),
                'transaction_ids': [tx.id for tx in processed_transactions],
                'errors': errors[:10]  # Limit error messages
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Database error: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/policies/upload', methods=['POST'])
def upload_policy():
    """Upload and process policy document"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get form data
        title = request.form.get('title', file.filename)
        jurisdiction = request.form.get('jurisdiction', 'US')
        
        # Read content
        content = file.read().decode('utf-8')
        
        # Generate policy ID
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        policy_id = f"pol_{content_hash[:16]}"
        
        # Check if already processed
        existing_policy = db.session.query(PolicyDocument).filter(
            PolicyDocument.document_hash == content_hash
        ).first()
        
        if existing_policy:
            # Return existing obligations
            obligations = db.session.query(Obligation).filter(
                Obligation.policy_document_id == existing_policy.id
            ).all()
            
            return jsonify({
                'success': True,
                'policy_id': existing_policy.id,
                'obligations': [format_obligation(obl) for obl in obligations],
                'cached': True
            })
        
        # Create new policy document
        policy_doc = PolicyDocument(
            id=policy_id,
            title=title,
            content=content,
            document_hash=content_hash,
            source='web_upload',
            jurisdiction=jurisdiction,
            is_processed=False
        )
        
        db.session.add(policy_doc)
        db.session.flush()
        
        # Extract obligations using Gemini
        obligations = asyncio.run(extract_obligations_async(content, policy_id))
        
        # Save obligations
        for obl_data in obligations:
            obligation = Obligation(
                id=obl_data['id'],
                policy_document_id=policy_doc.id,
                actor=obl_data['actor'],
                action=obl_data['action'],
                type=obl_data['type'],
                condition=obl_data.get('condition'),
                jurisdiction=obl_data.get('jurisdiction', jurisdiction),
                temporal_scope=obl_data.get('temporal_scope'),
                confidence=obl_data['confidence'],
                extraction_model='gemini-1.5-flash',
                source_clause=obl_data['source_clause']
            )
            db.session.add(obligation)
        
        policy_doc.is_processed = True
        db.session.commit()
        
        return jsonify({
            'success': True,
            'policy_id': policy_doc.id,
            'obligations': obligations
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/compliance/scan', methods=['POST'])
def run_compliance_scan():
    """Run compliance scan on transactions"""
    try:
        data = request.get_json()
        transaction_ids = data.get('transaction_ids', [])
        jurisdiction = data.get('jurisdiction', 'US')
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        # If no specific IDs provided, use recent transactions
        if not transaction_ids:
            recent_transactions = db.session.query(Transaction).limit(50).all()
            transaction_ids = [tx.id for tx in recent_transactions]
        
        # Fetch transactions
        transactions = db.session.query(Transaction).filter(
            Transaction.id.in_(transaction_ids)
        ).all()
        
        violations = []
        
        for transaction in transactions:
            # Convert to dict for rule checking
            tx_data = {
                'id': transaction.id,
                'amount_usd': transaction.amount_usd,
                'type': transaction.type,
                'amount_1d_total': transaction.amount_1d_total,
                'transaction_count_1d': transaction.transaction_count_1d
            }
            
            # Check compliance rules
            rule_violations = ComplianceRules.check_violations(tx_data, jurisdiction)
            
            for violation_data in rule_violations:
                # Generate violation with explanation
                explanation = asyncio.run(generate_violation_explanation_async(
                    tx_data, violation_data
                ))
                
                violation = {
                    'violation_id': f"viol_{uuid.uuid4().hex[:16]}",
                    'transaction_id': transaction.id,
                    'violation_type': violation_data['type'],
                    'confidence': violation_data['confidence'],
                    'risk_level': violation_data['risk_level'],
                    'reasoning': violation_data['reasoning'],
                    'recommended_action': violation_data['recommended_action'],
                    'human_explanation': explanation,
                    'evidence_ids': ['transaction_data', 'compliance_rule'],
                    'detected_at': datetime.utcnow().isoformat()
                }
                
                violations.append(violation)
                
                # Save to database
                db_violation = Violation(
                    id=violation['violation_id'],
                    transaction_id=transaction.id,
                    obligation_id=f"rule_{violation_data['type']}",
                    violation_type=violation_data['type'],
                    confidence=violation_data['confidence'],
                    risk_level=violation_data['risk_level'],
                    reasoning=violation_data['reasoning'],
                    recommended_action=violation_data['recommended_action'],
                    evidence_ids=['transaction_data', 'compliance_rule'],
                    llm_explanation=explanation,
                    model_version='gemini-1.5-flash'
                )
                db.session.add(db_violation)
        
        db.session.commit()
        
        # Generate summary
        summary = {
            'total_transactions': len(transactions),
            'total_violations': len(violations),
            'violation_rate': len(violations) / len(transactions) if transactions else 0,
            'risk_distribution': {},
            'violation_types': {}
        }
        
        # Count by risk level and type
        for violation in violations:
            risk = violation['risk_level']
            vtype = violation['violation_type']
            
            summary['risk_distribution'][risk] = summary['risk_distribution'].get(risk, 0) + 1
            summary['violation_types'][vtype] = summary['violation_types'].get(vtype, 0) + 1
        
        return jsonify({
            'success': True,
            'scan_id': f"scan_{int(datetime.now().timestamp())}",
            'transaction_count': len(transactions),
            'violation_count': len(violations),
            'violations': violations,
            'summary': summary
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/violations/recent')
def get_recent_violations():
    """Get recent violations"""
    try:
        limit = request.args.get('limit', 20, type=int)
        status = request.args.get('status')
        
        query = db.session.query(Violation).order_by(Violation.detected_at.desc())
        
        if status:
            query = query.filter(Violation.review_status == status)
        
        violations = query.limit(limit).all()
        
        return jsonify([{
            'id': v.id,
            'transaction_id': v.transaction_id,
            'violation_type': v.violation_type,
            'confidence': v.confidence,
            'risk_level': v.risk_level,
            'reasoning': v.reasoning,
            'recommended_action': v.recommended_action,
            'review_status': v.review_status,
            'detected_at': v.detected_at.isoformat() if v.detected_at else None,
            'human_explanation': v.llm_explanation
        } for v in violations])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/violations/<violation_id>/review', methods=['POST'])
def review_violation(violation_id):
    """Review and approve/reject violation"""
    try:
        data = request.get_json()
        action = data.get('action')  # 'approve' or 'reject'
        notes = data.get('notes', '')
        reviewer_id = data.get('reviewer_id', 'web_user')
        
        if action not in ['approve', 'reject']:
            return jsonify({'error': 'Invalid action'}), 400
        
        violation = db.session.query(Violation).filter(Violation.id == violation_id).first()
        if not violation:
            return jsonify({'error': 'Violation not found'}), 404
        
        violation.review_status = 'approved' if action == 'approve' else 'rejected'
        violation.review_notes = notes
        violation.reviewer_id = reviewer_id
        violation.reviewed_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Violation {action}d successfully'
        })
        
    except Exception as e:
        db.session.rollback()
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
def generate_sample_demo_data():
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