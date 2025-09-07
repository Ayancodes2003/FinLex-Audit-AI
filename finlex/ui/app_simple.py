"""
FinLex Audit AI - Simplified Streamlit Dashboard (Demo Version)

A lightweight dashboard for testing without full database setup.
"""

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import os
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="FinLex Audit AI - Demo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple API client for testing
class MockFinLexAPI:
    """Mock API client for demo purposes"""
    
    def get_dashboard_stats(self):
        """Mock dashboard statistics"""
        return {
            "total_transactions": 1234,
            "total_policies": 15,
            "total_violations": 28,
            "pending_reviews": 5,
            "recent_violations": 8
        }
    
    def upload_transactions(self, file_content, filename):
        """Mock transaction upload"""
        try:
            # Parse CSV to count rows
            import io
            df = pd.read_csv(io.BytesIO(file_content))
            return {
                "success": True,
                "processed_count": len(df),
                "failed_count": 0,
                "transaction_ids": [f"tx_{i:06d}" for i in range(len(df))]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def upload_policy(self, file_content, filename, **kwargs):
        """Mock policy upload"""
        try:
            content_str = file_content.decode('utf-8')
            # Mock extracted obligations
            obligations = [
                {
                    "id": "obl_001",
                    "actor": "financial institutions",
                    "action": "report transactions exceeding $10,000",
                    "type": "requirement",
                    "condition": "transaction amount > $10,000",
                    "jurisdiction": "US",
                    "temporal_scope": "within 15 days",
                    "confidence": 0.95,
                    "source_clause": content_str[:200] + "..."
                },
                {
                    "id": "obl_002",
                    "actor": "banks",
                    "action": "maintain transaction records",
                    "type": "requirement", 
                    "condition": "all wire transfers",
                    "jurisdiction": "US",
                    "temporal_scope": "5 years",
                    "confidence": 0.88,
                    "source_clause": content_str[200:400] + "..."
                }
            ]
            return {
                "success": True,
                "policy_id": "pol_demo_001",
                "obligations": obligations
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_compliance_scan(self, **kwargs):
        """Mock compliance scan"""
        transaction_count = len(kwargs.get("transaction_ids", []))
        
        # Generate mock violations based on transaction count
        violations = []
        if transaction_count > 0:
            violations = [
                {
                    "transaction_id": "tx_000001",
                    "obligation_id": "obl_001",
                    "violation_type": "large_transaction",
                    "confidence": 0.95,
                    "risk_level": "high",
                    "reasoning": "Transaction amount $150,000 exceeds large transaction threshold of $100,000",
                    "recommended_action": "escalate",
                    "evidence_ids": ["transaction_data", "threshold_rule"],
                    "human_explanation": "This transaction requires immediate review due to its size exceeding regulatory thresholds."
                },
                {
                    "transaction_id": "tx_000002", 
                    "obligation_id": "obl_001",
                    "violation_type": "cash_reporting",
                    "confidence": 0.87,
                    "risk_level": "medium",
                    "reasoning": "Cash transaction of $15,000 exceeds reporting threshold",
                    "recommended_action": "review",
                    "evidence_ids": ["transaction_data", "cash_rule"],
                    "human_explanation": "Cash transaction requires regulatory reporting within 24 hours."
                }
            ]
        
        return {
            "success": True,
            "scan_id": f"scan_{int(datetime.now().timestamp())}",
            "transaction_count": transaction_count,
            "violation_count": len(violations),
            "violations": violations,
            "summary": {
                "total_transactions": transaction_count,
                "clean_transactions": transaction_count - len(violations),
                "violation_rate": len(violations) / max(transaction_count, 1),
                "risk_distribution": {"high": 1, "medium": 1, "low": 0},
                "violation_types": {"large_transaction": 1, "cash_reporting": 1}
            },
            "processing_time_seconds": 2.5
        }

# Initialize mock API
api = MockFinLexAPI()

def main():
    """Main application"""
    st.sidebar.title("🏦 FinLex Audit AI")
    st.sidebar.markdown("*Demo Version - Mock Data*")
    
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Upload Data", "Policy Management", "Compliance Scan"]
    )
    
    if page == "Dashboard":
        render_dashboard()
    elif page == "Upload Data":
        render_upload_data()
    elif page == "Policy Management":
        render_policy_management()
    elif page == "Compliance Scan":
        render_compliance_scan()

def render_dashboard():
    """Render main dashboard"""
    st.title("📊 Compliance Dashboard")
    st.markdown("*Demo showing mock compliance data*")
    
    # Get mock stats
    stats = api.get_dashboard_stats()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{stats['total_transactions']:,}")
    with col2:
        st.metric("Active Policies", f"{stats['total_policies']:,}")
    with col3:
        st.metric("Total Violations", f"{stats['total_violations']:,}", 
                 delta=f"+{stats['recent_violations']} this week")
    with col4:
        st.metric("Pending Reviews", f"{stats['pending_reviews']:,}")
    
    st.divider()
    
    # Mock recent violations
    st.subheader("🚨 Recent Violations")
    
    violations_data = [
        {"ID": "viol_001", "Type": "Large Transaction", "Risk": "High", "Status": "Pending"},
        {"ID": "viol_002", "Type": "Cash Reporting", "Risk": "Medium", "Status": "Reviewed"},
        {"ID": "viol_003", "Type": "Velocity Limit", "Risk": "High", "Status": "Pending"},
        {"ID": "viol_004", "Type": "Frequency Limit", "Risk": "Low", "Status": "Approved"}
    ]
    
    df = pd.DataFrame(violations_data)
    st.dataframe(df, use_container_width=True)
    
    # System status
    st.divider()
    st.success("✅ Demo system operational")

def render_upload_data():
    """Render data upload interface"""
    st.title("📤 Upload Transaction Data")
    st.markdown("*Upload CSV files with transaction data*")
    
    uploaded_file = st.file_uploader(
        "Choose transaction CSV file",
        type=['csv'],
        help="Upload CSV file with transaction data"
    )
    
    if uploaded_file is not None:
        try:
            # Show file preview
            df = pd.read_csv(uploaded_file)
            st.write("**File Preview:**")
            st.dataframe(df.head())
            
            st.write(f"**File Info:** {len(df)} rows, {len(df.columns)} columns")
            
            # Upload button
            if st.button("Upload Transactions", type="primary"):
                with st.spinner("Processing transaction data..."):
                    # Reset file pointer and get bytes
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    
                    result = api.upload_transactions(file_bytes, uploaded_file.name)
                    
                    if result.get("success"):
                        st.success(f"""
                        ✅ **Upload Successful!**
                        
                        - Processed: {result.get('processed_count', 0)} transactions
                        - Generated Transaction IDs: {len(result.get('transaction_ids', []))}
                        """)
                        
                        # Show some sample transaction IDs
                        if result.get('transaction_ids'):
                            st.write("**Sample Transaction IDs:**")
                            sample_ids = result['transaction_ids'][:5]
                            for tid in sample_ids:
                                st.code(tid)
                    else:
                        st.error(f"Upload failed: {result.get('error')}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Show expected format
    st.divider()
    st.subheader("📋 Expected CSV Format")
    
    sample_data = {
        "step": [1, 2, 3],
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
        "amount": [150000, 15000, 5000],
        "nameOrig": ["C1234567890", "C2345678901", "C3456789012"],
        "nameDest": ["M9876543210", "M8765432109", "M7654321098"],
        "isFraud": [0, 0, 0]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)

def render_policy_management():
    """Render policy management interface"""
    st.title("📋 Policy Management")
    st.markdown("*Upload and analyze policy documents*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_policy = st.file_uploader(
            "Choose policy document",
            type=['txt', 'md'],
            help="Upload text or markdown policy document"
        )
    
    with col2:
        title = st.text_input("Policy Title", placeholder="Enter policy title")
        jurisdiction = st.selectbox("Jurisdiction", ["US", "EU", "UK"])
    
    if uploaded_policy is not None:
        try:
            # Show content preview
            content = uploaded_policy.read().decode('utf-8')
            st.write("**Policy Content Preview:**")
            st.text_area("Content", content[:500] + "...", height=200, disabled=True)
            
            # Process button
            if st.button("Process Policy Document", type="primary"):
                with st.spinner("Extracting obligations from policy..."):
                    uploaded_policy.seek(0)
                    file_content = uploaded_policy.read()
                    
                    result = api.upload_policy(
                        file_content,
                        uploaded_policy.name,
                        title=title,
                        jurisdiction=jurisdiction
                    )
                    
                    if result.get("success"):
                        st.success("✅ **Policy Processed Successfully!**")
                        
                        obligations = result.get("obligations", [])
                        st.write(f"**Extracted {len(obligations)} obligations:**")
                        
                        # Display extracted obligations
                        for i, obligation in enumerate(obligations):
                            with st.expander(f"Obligation {i+1}: {obligation.get('actor', 'Unknown')}"):
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.write(f"**Action:** {obligation.get('action')}")
                                    st.write(f"**Type:** {obligation.get('type')}")
                                    st.write(f"**Condition:** {obligation.get('condition')}")
                                
                                with col_b:
                                    st.write(f"**Jurisdiction:** {obligation.get('jurisdiction')}")
                                    st.write(f"**Confidence:** {obligation.get('confidence', 0):.1%}")
                                    st.write(f"**Temporal Scope:** {obligation.get('temporal_scope')}")
                                
                                st.write(f"**Source:** {obligation.get('source_clause')}")
                    else:
                        st.error(f"Processing failed: {result.get('error')}")
        except UnicodeDecodeError:
            st.error("Unable to decode file. Please ensure it's UTF-8 encoded.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def render_compliance_scan():
    """Render compliance scanning interface"""
    st.title("🔍 Compliance Scan")
    st.markdown("*Run compliance analysis on transactions*")
    
    # Scan configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tx_count = st.slider("Number of transactions to scan", 1, 100, 25)
        jurisdiction = st.selectbox("Jurisdiction", ["US", "EU", "UK"])
    
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7)
        st.info(f"📊 **Scan Summary:**\n- {tx_count} transactions\n- Jurisdiction: {jurisdiction}")
    
    # Generate demo transaction IDs
    transaction_ids = [f"tx_{i:06d}" for i in range(tx_count)]
    
    if st.button("🚀 Run Compliance Scan", type="primary"):
        with st.spinner(f"Scanning {len(transaction_ids)} transactions..."):
            # Simulate progress
            progress_bar = st.progress(0)
            import time
            for i in range(10):
                time.sleep(0.1)
                progress_bar.progress((i + 1) / 10)
            
            # Run mock scan
            result = api.run_compliance_scan(
                transaction_ids=transaction_ids,
                jurisdiction=jurisdiction,
                confidence_threshold=confidence_threshold
            )
            
            progress_bar.progress(1.0)
        
        # Display results
        if result.get("success"):
            st.success("✅ **Compliance Scan Completed!**")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Transactions Scanned", result["transaction_count"])
            with col2:
                st.metric("Violations Found", result["violation_count"])
            with col3:
                rate = result["summary"]["violation_rate"] * 100
                st.metric("Violation Rate", f"{rate:.1f}%")
            with col4:
                st.metric("Processing Time", f"{result['processing_time_seconds']:.1f}s")
            
            # Violations details
            violations = result.get("violations", [])
            if violations:
                st.subheader("🚨 Detected Violations")
                
                for violation in violations:
                    risk_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    risk_icon = risk_color.get(violation.get("risk_level", "medium"), "⚪")
                    
                    with st.expander(f"{risk_icon} {violation.get('violation_type', 'Unknown')} - {violation.get('risk_level', 'medium').upper()} Risk"):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.write(f"**Transaction ID:** {violation.get('transaction_id')}")
                            st.write(f"**Confidence:** {violation.get('confidence', 0):.1%}")
                            st.write(f"**Reasoning:** {violation.get('reasoning')}")
                            st.write(f"**Explanation:** {violation.get('human_explanation')}")
                        
                        with col_b:
                            st.write(f"**Recommended Action:** {violation.get('recommended_action').title()}")
                            st.write(f"**Evidence Count:** {len(violation.get('evidence_ids', []))}")
                            
                            # Action buttons
                            col_x, col_y = st.columns(2)
                            with col_x:
                                if st.button(f"✅ Approve", key=f"approve_{violation.get('transaction_id')}"):
                                    st.success("Violation approved!")
                            with col_y:
                                if st.button(f"❌ Reject", key=f"reject_{violation.get('transaction_id')}"):
                                    st.info("Violation rejected!")
            else:
                st.info("🎉 No violations detected in the scanned transactions!")

# Load sample data if files exist
def load_sample_files():
    """Load sample data files if they exist"""
    sample_files = []
    
    if Path("sample_transactions.csv").exists():
        sample_files.append("📄 sample_transactions.csv - Ready for upload")
    
    if Path("sample_policy.txt").exists():
        sample_files.append("📄 sample_policy.txt - Ready for upload")
    
    if sample_files:
        st.sidebar.markdown("### 📁 Sample Files Available:")
        for file in sample_files:
            st.sidebar.markdown(f"- {file}")

if __name__ == "__main__":
    # Load sample files info
    load_sample_files()
    
    # Run main app
    main()