#!/usr/bin/env python3
"""
FinLex Audit AI Setup Script

Quick setup and validation script for the FinLex compliance system.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path


def print_banner():
    """Print FinLex banner"""
    print("""
    ╔═══════════════════════════════════════════════╗
    ║              🏦 FinLex Audit AI               ║
    ║         AI-Powered Compliance Analysis        ║
    ╚═══════════════════════════════════════════════╝
    """)


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")
    return True


def check_docker():
    """Check Docker installation"""
    print("🐳 Checking Docker...")
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Docker not found. Please install Docker Desktop.")
    return False


def check_environment():
    """Check environment variables"""
    print("🔧 Checking environment configuration...")
    
    env_file = Path('finlex/.env')
    if not env_file.exists():
        print("⚠️  .env file not found. Copying from example...")
        example_file = Path('finlex/infra/.env.example')
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            print("📄 .env file created. Please edit it with your API keys.")
        else:
            print("❌ .env.example not found")
            return False
    
    # Check for required environment variables
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    if 'GEMINI_API_KEY=your_gemini_api_key_here' in env_content:
        print("⚠️  Please update GEMINI_API_KEY in .env file")
        return False
    
    print("✅ Environment configuration looks good")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    try:
        # Install main dependencies
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'finlex/requirements.txt'
        ], check=True, capture_output=True)
        
        # Install UI dependencies
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'finlex/requirements_ui.txt'
        ], check=True, capture_output=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def start_services():
    """Start services using Docker Compose"""
    print("🚀 Starting FinLex services...")
    
    try:
        os.chdir('finlex/infra')
        
        # Start services
        subprocess.run([
            'docker-compose', 'up', '-d'
        ], check=True)
        
        print("⏳ Waiting for services to start...")
        time.sleep(30)  # Give services time to start
        
        os.chdir('../..')  # Return to root directory
        print("✅ Services started successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}")
        return False


def check_service_health():
    """Check if all services are healthy"""
    print("🏥 Checking service health...")
    
    services = [
        ("Main API", "http://localhost:8000/health"),
        ("Streamlit UI", "http://localhost:8501/_stcore/health"),
    ]
    
    all_healthy = True
    
    for service_name, url in services:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {service_name} is healthy")
            else:
                print(f"⚠️  {service_name} returned status {response.status_code}")
                all_healthy = False
        except requests.exceptions.RequestException as e:
            print(f"❌ {service_name} is not responding: {e}")
            all_healthy = False
    
    return all_healthy


def run_basic_tests():
    """Run basic functionality tests"""
    print("🧪 Running basic tests...")
    
    try:
        os.chdir('finlex')
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/test_rules.py::TestTransactionProcessor::test_currency_normalization', '-v'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Basic tests passed")
            return True
        else:
            print(f"❌ Tests failed: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False
    finally:
        os.chdir('..')


def print_success_message():
    """Print success message with next steps"""
    print("""
    🎉 FinLex Audit AI setup completed successfully!

    📊 Dashboard: http://localhost:8501
    🔌 API Docs: http://localhost:8000/docs
    
    🚀 Next Steps:
    1. Open the dashboard at http://localhost:8501
    2. Upload sample transaction data in the "Upload Data" section
    3. Upload policy documents in the "Policy Management" section
    4. Run compliance scans in the "Compliance Scan" section
    
    📚 Documentation:
    - README.md - Complete system overview
    - finlex/docs/runbook.md - Operations guide
    
    🛠️  Troubleshooting:
    - Check service logs: docker-compose logs -f
    - Restart services: docker-compose restart
    - Stop services: docker-compose down
    
    Happy compliance monitoring! 🏦✨
    """)


def main():
    """Main setup function"""
    print_banner()
    
    # Check prerequisites
    checks = [
        check_python_version(),
        check_docker(),
        check_environment(),
    ]
    
    if not all(checks):
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Install and start
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        sys.exit(1)
    
    if not start_services():
        print("\n❌ Failed to start services.")
        sys.exit(1)
    
    # Validate
    if not check_service_health():
        print("\n⚠️  Some services may not be fully healthy. Check logs with: docker-compose logs")
    
    if run_basic_tests():
        print_success_message()
    else:
        print("\n⚠️  Setup completed but some tests failed. System should still be functional.")
        print("📊 Dashboard: http://localhost:8501")


if __name__ == "__main__":
    main()