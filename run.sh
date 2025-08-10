#!/bin/bash

# Solar Panel Detector Setup Script

echo "Setting up Solar Panel Detector..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # Linux/Mac
    source .venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create static directory if it doesn't exist
mkdir -p static

echo "Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source .venv/Scripts/activate"
else
    echo "   source .venv/bin/activate"
fi
echo "2. Start the server:"
echo "   uvicorn app:app --reload"
echo "3. Open http://127.0.0.1:8000 in your browser"
