#!/bin/bash
set -e  # stop if any script fails

echo "Starting pipeline..."

echo "Fetching stock prices..."
python3 get_stocks.py

echo "Processing and scoring..."
python3 processing.py

echo "Generating graphs..."
python3 graphs.py

echo "Pipeline complete!"