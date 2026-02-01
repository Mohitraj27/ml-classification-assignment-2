#!/bin/bash
# Diagnostic script for ML Assignment setup

echo "==================================="
echo "ML Assignment 2 - System Check"
echo "==================================="

echo -e "\n1. Checking Python..."
if command -v python3 &> /dev/null; then
    python3 --version
    echo "✅ Python3 found"
else
    echo "❌ Python3 not found"
fi

echo -e "\n2. Checking pip..."
if command -v pip3 &> /dev/null; then
    pip3 --version
    echo "✅ pip3 found"
else
    echo "❌ pip3 not found"
fi

echo -e "\n3. Checking virtual environment..."
if [ -d "venv" ]; then
    echo "✅ venv directory exists"
else
    echo "❌ venv directory not found"
    echo "Run: python3 -m venv venv"
fi

echo -e "\n4. Checking if venv is active..."
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtual environment is active"
    echo "Location: $VIRTUAL_ENV"
else
    echo "❌ Virtual environment not active"
    echo "Run: source venv/bin/activate"
fi

echo -e "\n5. Checking required files..."
files=("app.py" "requirements.txt" "README.md")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file not found"
    fi
done

echo -e "\n6. Checking installed packages..."
if [ -n "$VIRTUAL_ENV" ]; then
    if pip list | grep -q streamlit; then
        echo "✅ streamlit installed"
    else
        echo "❌ streamlit not installed"
        echo "Run: pip install -r requirements.txt"
    fi
else
    echo "⚠️  Activate virtual environment first"
fi

echo -e "\n==================================="
echo "System Check Complete"
echo "==================================="
