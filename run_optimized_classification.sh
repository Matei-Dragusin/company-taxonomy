#!/bin/bash
# Script to run the full optimized classification pipeline

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p results
mkdir -p logs
mkdir -p optimization_results

echo "===== STARTING INSURANCE TAXONOMY CLASSIFICATION PIPELINE ====="
echo ""

# Check if input files exist
if [ ! -f "data/raw/companies.csv" ]; then
    echo "Error: companies.csv not found in data/raw directory"
    echo "Please place the companies file in data/raw directory"
    exit 1
fi

if [ ! -f "data/raw/insurance_taxonomy.csv" ]; then
    echo "Error: insurance_taxonomy.csv not found in data/raw directory"
    echo "Please place the taxonomy file in data/raw directory"
    exit 1
fi

# Step 1: Preprocess data
echo "Step 1: Preprocessing data..."
python run_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi
echo "Preprocessing completed successfully"
echo ""

# Step 2: Run hyperparameter optimization
echo "Step 2: Running hyperparameter optimization..."
echo "This may take some time, please be patient..."
python hyperparameter_optimizer.py --n-trials 30 --timeout 1800
if [ $? -ne 0 ]; then
    echo "Error: Hyperparameter optimization failed"
    echo "Continuing with default parameters"
else
    echo "Hyperparameter optimization completed successfully"
fi
echo ""

# Step 3: Run classification with optimized parameters
echo "Step 3: Running classification with optimized parameters..."
python run_classification.py --use-optimized-params --evaluate --output-file "optimized_classified_companies.csv" --description-label-file "optimized_description_label_results.csv"
if [ $? -ne 0 ]; then
    echo "Error: Classification failed"
    exit 1
fi
echo "Classification completed successfully"
echo ""

# Step 4: Check results
echo "Step 4: Checking results..."
if [ -f "data/processed/optimized_description_label_results.csv" ]; then
    echo "Output file: data/processed/optimized_description_label_results.csv"
    echo "Sample of the results (first 5 rows):"
    head -n 5 "data/processed/optimized_description_label_results.csv"
else
    echo "Error: Output file not found"
fi
echo ""

echo "===== CLASSIFICATION PIPELINE COMPLETED ====="
echo "All results are available in the data/processed directory"
