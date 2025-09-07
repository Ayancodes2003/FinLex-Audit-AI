#!/bin/bash

# FinLex Audit AI - Web Application Startup Script

echo "🏦 Starting FinLex Audit AI Web Application..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your actual configuration values"
    echo "🔑 Don't forget to add your GEMINI_API_KEY!"
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p uploads logs ssl

# Start services with Docker Compose
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

echo ""
echo "✅ FinLex Audit AI is starting up!"
echo "🌐 Web Application: http://localhost:5000"
echo "🗄️  Database: PostgreSQL on port 5432"
echo "🔴 Redis: Available on port 6379"
echo "📊 Nginx Proxy: Available on port 80"
echo ""
echo "📝 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
echo "🔄 To restart: docker-compose restart"