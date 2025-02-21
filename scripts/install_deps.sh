#!/bin/bash

# Exit on error
set -e

echo "Setting up Vigilare development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create example config files if they don't exist
if [ ! -f "config/config.yaml" ]; then
    echo "Creating config files..."
    cp config/config.yaml.example config/config.yaml
    cp config/logging_config.yaml.example config/logging_config.yaml
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/screenshots
mkdir -p data/vectors
mkdir -p data/logs

# Setup database
echo "Setting up database..."
python scripts/setup_db.py

echo "Setup completed successfully!" 