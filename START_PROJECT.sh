#!/bin/bash
# Script to start HealthPodcasIQ project

echo "🚀 Starting HealthPodcasIQ Project"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "models" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 backend/check_dependencies.py
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Some dependencies may be missing. Installing..."
    cd backend
    pip3 install -r requirements.txt
    cd ..
fi

echo ""
echo "🔍 Testing imports..."
python3 backend/test_imports.py
if [ $? -ne 0 ]; then
    echo "❌ Import test failed. Please check the errors above."
    exit 1
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "Starting backend server..."
echo "Backend will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start backend
cd backend
python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

