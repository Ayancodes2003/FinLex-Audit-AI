"""
FinLex Audit AI - Streamlit Dashboard

A comprehensive dashboard for AI-powered financial compliance analysis.
"""

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.express as px
from typing import Dict, List, Any
import io

# Configure Streamlit page
st.set_page_config(
    page_title="FinLex Audit AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

class FinLexAPI:
    """API client for FinLex backend services"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        try:
            response = self.session.get(f"{self.base_url}/dashboard/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch stats: {str(e)}")
            return {}
    
    def upload_transactions(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload transaction CSV file"""
        try:
            files = {"file": (filename, file_content, "text/csv")}
            response = self.session.post(f"{self.base_url}/transactions/upload", files=files)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def upload_policy(self, file_content: bytes, filename: str, **kwargs) -> Dict[str, Any]:
        """Upload policy document"""
        try:
            files = {"file": (filename, file_content, "text/plain")}
            response = self.session.post(f"{self.base_url}/policies/upload", files=files, params=kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_compliance_scan(self, **kwargs) -> Dict[str, Any]:
        """Run compliance scan"""
        try:
            response = self.session.post(f"{self.base_url}/scan", json=kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

@st.cache_resource
def get_api_client():
    return FinLexAPI()

api = get_api_client()

def main():
    """Main application"""
    st.sidebar.title("🏦 FinLex Audit AI")
    
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
    
    with st.spinner("Loading dashboard data..."):
        stats = api.get_dashboard_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{stats.get('total_transactions', 0):,}")
        with col2:
            st.metric("Active Policies", f"{stats.get('total_policies', 0):,}")
        with col3:
            st.metric("Total Violations", f"{stats.get('total_violations', 0):,}")
        with col4:
            st.metric("Pending Reviews", f"{stats.get('pending_reviews', 0):,}")
    
    st.info("Dashboard loaded successfully!")

def render_upload_data():
    """Render data upload interface"""
    st.title("📤 Upload Transaction Data")
    
    uploaded_file = st.file_uploader("Choose transaction CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**File Preview:**")
            st.dataframe(df.head())
            
            if st.button("Upload Transactions", type="primary"):
                with st.spinner("Processing..."):
                    uploaded_file.seek(0)
                    result = api.upload_transactions(uploaded_file.read(), uploaded_file.name)
                    
                    if result.get("success"):
                        st.success(f"✅ Uploaded {result.get('processed_count', 0)} transactions!")
                    else:
                        st.error(f"Upload failed: {result.get('error')}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def render_policy_management():
    """Render policy management interface"""
    st.title("📋 Policy Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_policy = st.file_uploader("Choose policy document", type=['txt', 'md'])
    
    with col2:
        title = st.text_input("Policy Title")
        jurisdiction = st.selectbox("Jurisdiction", ["US", "EU", "UK"])
    
    if uploaded_policy is not None:
        content = uploaded_policy.read().decode('utf-8')
        st.text_area("Content Preview", content[:500] + "...", height=200, disabled=True)
        
        if st.button("Process Policy", type="primary"):
            with st.spinner("Extracting obligations..."):
                uploaded_policy.seek(0)
                result = api.upload_policy(
                    uploaded_policy.read(), 
                    uploaded_policy.name,
                    title=title,
                    jurisdiction=jurisdiction
                )
                
                if result.get("success"):
                    st.success("✅ Policy processed successfully!")
                    obligations = result.get("obligations", [])
                    st.write(f"Extracted {len(obligations)} obligations")
                else:
                    st.error(f"Processing failed: {result.get('error')}")

def render_compliance_scan():
    """Render compliance scanning interface"""
    st.title("🔍 Compliance Scan")
    
    # Simple transaction ID input for demo
    tx_count = st.slider("Number of transactions to scan", 10, 100, 50)
    jurisdiction = st.selectbox("Jurisdiction", ["US", "EU", "UK"])
    
    # Generate demo transaction IDs
    transaction_ids = [f"tx_{i:06d}" for i in range(tx_count)]
    
    if st.button("🚀 Run Compliance Scan", type="primary"):
        with st.spinner(f"Scanning {len(transaction_ids)} transactions..."):
            result = api.run_compliance_scan(
                transaction_ids=transaction_ids,
                jurisdiction=jurisdiction,
                generate_explanations=True,
                confidence_threshold=0.7
            )
            
            if result.get("success"):
                st.success("✅ Scan completed!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Scanned", result.get("transaction_count", 0))
                with col2:
                    st.metric("Violations", result.get("violation_count", 0))
                with col3:
                    rate = (result.get("violation_count", 0) / result.get("transaction_count", 1)) * 100
                    st.metric("Violation Rate", f"{rate:.1f}%")
                
                violations = result.get("violations", [])
                if violations:
                    st.subheader("🚨 Detected Violations")
                    for violation in violations[:5]:  # Show first 5
                        with st.expander(f"Violation: {violation.get('violation_type', 'Unknown')}"):
                            st.write(f"**Risk:** {violation.get('risk_level', 'medium').upper()}")
                            st.write(f"**Confidence:** {violation.get('confidence', 0):.1%}")
                            st.write(f"**Reasoning:** {violation.get('reasoning', 'N/A')}")
                else:
                    st.info("🎉 No violations detected!")
            else:
                st.error(f"Scan failed: {result.get('error')}")

if __name__ == "__main__":
    main()