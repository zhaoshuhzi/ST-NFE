#!/bin/bash

echo "Running ST-NFE Demo..."

# Check if Python is installed
if ! command -v python &> /dev/null
then
    echo "❌ Error: Python is not found. Please install Python 3.8+."
    exit
fi

# Install dependencies (optional, real setup should do this manually)
# pip install -r requirements.txt

# Run main script in mock mode
python main.py --mode mock

echo "========================================"
echo "   Demo Complete. Check main.py for details."
echo "========================================"
