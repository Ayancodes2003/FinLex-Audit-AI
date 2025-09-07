#!/usr/bin/env python3
"""
Simple validation script to check the FinLex codebase structure and imports.
"""

import os
import sys
from pathlib import Path

def validate_structure():
    """Validate project structure"""
    print("🔍 Validating project structure...")
    
    expected_files = [
        'finlex/services/gemini_client.py',
        'finlex/services/database.py',
        'finlex/services/ingest/main.py',
        'finlex/services/extractor/main.py',
        'finlex/services/matcher/main.py',
        'finlex/services/raggenerator/main.py',
        'finlex/services/api/main.py',
        'finlex/ui/app.py',
        'finlex/infra/docker-compose.yml',
        'finlex/tests/test_rules.py',
        'finlex/requirements.txt',
        'finlex/requirements_ui.txt',
        'finlex/docs/runbook.md'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def validate_syntax():
    """Basic syntax validation"""
    print("🐍 Validating Python syntax...")
    
    python_files = [
        'finlex/services/gemini_client.py',
        'finlex/services/database.py',
        'finlex/ui/app.py',
        'setup.py'
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic syntax check
            compile(content, file_path, 'exec')
            print(f"✅ {file_path} - syntax OK")
        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            return False
        except Exception as e:
            print(f"⚠️  {file_path} - warning: {e}")
    
    return True

def check_requirements():
    """Check requirements files"""
    print("📦 Checking requirements files...")
    
    req_files = ['finlex/requirements.txt', 'finlex/requirements_ui.txt']
    
    for req_file in req_files:
        if Path(req_file).exists():
            with open(req_file, 'r') as f:
                lines = f.readlines()
            print(f"✅ {req_file} - {len(lines)} dependencies")
        else:
            print(f"❌ {req_file} - not found")
            return False
    
    return True

def show_getting_started():
    """Show getting started instructions"""
    print("""
    🚀 FinLex Audit AI - Ready for Installation!
    
    📋 To get started:
    
    1. Install Python dependencies:
       python -m venv venv
       venv\\Scripts\\activate  # On Windows
       # source venv/bin/activate  # On macOS/Linux
       
       cd finlex
       pip install -r requirements.txt
       pip install -r requirements_ui.txt
    
    2. Set up environment:
       copy infra\\.env.example .env
       # Edit .env and add your GEMINI_API_KEY
    
    3. Start with Docker (Recommended):
       cd infra
       docker-compose up -d
    
    4. Or start manually:
       # Terminal 1: API
       uvicorn services.api.main:app --port 8000
       
       # Terminal 2: UI
       streamlit run ui/app.py --server.port 8501
    
    5. Access the dashboard:
       http://localhost:8501
    
    📚 See README.md and docs/runbook.md for detailed instructions.
    """)

def main():
    """Main validation"""
    print("""
    ╔═══════════════════════════════════════════════╗
    ║         🏦 FinLex Audit AI Validator          ║
    ║            Pre-Installation Check             ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    checks = [
        validate_structure(),
        validate_syntax(),
        check_requirements()
    ]
    
    if all(checks):
        print("\n🎉 All validation checks passed!")
        show_getting_started()
    else:
        print("\n❌ Some validation checks failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())