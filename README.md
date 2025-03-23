# Insurance Company Taxonomy Classifier

This project implements a robust company classifier for an insurance taxonomy. It uses an ensemble approach combining TF-IDF, WordNet similarity, and keyword matching to accurately classify companies into insurance categories.

## Features

- Automatically classifies companies into insurance taxonomy categories
- Ensures every company gets at least one relevant tag
- Optimizes for close matching between company descriptions and assigned tags
- Uses multiple similarity metrics combined in an ensemble for better accuracy
- Includes hyperparameter optimization to fine-tune the classification parameters
- Exports results in CSV format with company descriptions and assigned tags

## Requirements

- Python 3.7+
- Required packages: See `requirements.txt`

## Setup

1. Clone the repository
2. Install dependencies:

   ```python
   pip install -r requirements.txt
   ```

3. Place your input files in the `data/raw` directory:
   - `companies.csv`: List of companies to classify
   - `insurance_taxonomy.csv`: Insurance taxonomy labels

## Directory Structure

```bash
/
├── data/
│   ├── raw/             # Raw input data
│   └── processed/       # Processed data and results
├── models/              # Saved models
├── results/             # Evaluation results
├── logs/                # Log files
├── optimization_results/# Hyperparameter optimization results
├── src/                 # Source code
│   ├── preprocessing/   # Data preprocessing modules
│   ├── feature_engineering/ # Feature engineering modules
│   └── ensemble/        # Ensemble classification modules
├── run_preprocessing.py # Script to run preprocessing
├── run_classification.py # Script to run classification
├── hyperparameter_optimizer.py # Script for parameter optimization
└── requirements.txt     # Python dependencies
```

## Usage

### Automatic Pipeline

The easiest way to run the complete pipeline is to use the provided shell script:

```bash
chmod +x run_optimized_classification.sh
./run_optimized_classification.sh
```

This will:

1. Preprocess the data
2. Run hyperparameter optimization
3. Run classification with optimized parameters
4. Output the results to `data/processed/optimized_description_label_results.csv`

### Manual Execution

Alternatively, you can run each step manually:

1. **Preprocessing:**

   ```python
   python run_preprocessing.py
   ```

2. **Hyperparameter Optimization:**

   ```python
   python hyperparameter_optimizer.py --n-trials 30 --timeout 1800
   ```

3. **Classification:**

   ```python
   python run_classification.py --use-optimized-params --evaluate
   ```

## Customization

You can customize various aspects of the classification process:

### Command Line Arguments

- `--top-k`: Maximum number of labels to assign to a company (default: 3)
- `--threshold`: Minimum similarity threshold to assign a label (default: 0.05)
- `--tfidf-weight`: Weight for TF-IDF similarity (default: 0.6)
- `--wordnet-weight`: Weight for WordNet similarity (default: 0.25)
- `--keyword-weight`: Weight for keyword similarity (default: 0.15)
- `--ensure-one-tag`: Ensure each company has at least one tag (default: True)
- `--use-optimized-params`: Use parameters from hyperparameter optimization
- `--evaluate`: Run evaluation after classification

Example:

```python
python run_classification.py --top-k 2 --threshold 0.07 --tfidf-weight 0.7 --ensure-one-tag --evaluate
```

## Output Format

The main output file `optimized_description_label_results.csv` contains:

- `description`: Company description
- `insurance_label`: Assigned insurance tags (comma-separated)
- `sector`: Company sector (if available)
- `niche`: Company niche (if available)
- `category`: Company category (if available)
- `label_scores`: Confidence scores for assigned tags (if requested)

## Evaluation

The system evaluates classification results based on:

- Coverage: Percentage of companies with at least one tag
- Average labels per company: Optimal is 1-2 labels per company
- Text similarity: How well tags match company descriptions
- Tag diversity: Range of different tags used across all companies

## Optimization

The hyperparameter optimization focuses on:

- Matching descriptions to tags (high priority)
- Confidence in assigned tags (high priority)
- Taxonomy coverage (medium priority)
- Label count per company (low priority)

## Further Development

Potential areas for improvement:

- Add validation with manually labeled data
- Implement more sophisticated NLP techniques
- Add support for cross-lingual classification
- Improve runtime efficiency for very large datasets
