#!/bin/bash
set -e  # stop if any script fails

echo "Starting pipeline..."

echo "Processing and scoring..."
python3 processing.py

echo "Fetching stock prices..."
python3 get_stocks.py

echo "Generating graphs..."
python3 graphs.py

echo "Analysis results..."
python3 analysis.py

echo "Running Machine Learning..."
python3 ml.py

echo "Pipeline complete!"