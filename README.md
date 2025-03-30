# Chained Email Classifier – Multi-Label AI Architecture

This project implements a Chained Multi-Output Architecture to classify customer support emails into multiple label layers: Type 2, Type 3, and Type 4.

## Project Structure

email_classifier/
├── main.py                  - Main controller script  
├── config.py                - Configuration settings  
├── requirements.txt         - Project dependencies  
│  
├── preprocessing/  
│   ├── data_loader.py       - Loads raw data  
│   ├── text_vectorizer.py   - TF-IDF vectorizer  
│   └── label_filter.py      - Filters out rare labels  
│  
├── models/  
│   ├── base_model.py        - Base class for ML models  
│   └── random_forest_model.py - Implements Random Forest  
│  
├── data_objects/  
│   └── encapsulated_data.py - Bundles training/test sets  
│  
├── evaluation/  
│   └── evaluator.py         - Metrics and reports  
│  
└── chained_model_manager.py - Core logic for chained predictions

## Chained Multi-Output Approach

This approach improves contextual classification by feeding one model’s output into the next.

1. Model 1 (Type 2): Predicts the primary category.
2. Model 2 (Type 3): Uses X + Type 2 predictions as input.
3. Model 3 (Type 4): Uses X + Type 2 + Type 3 predictions as input.

## Dataset Format

Place your dataset file in the root directory with the name: AppGallery.csv

Expected Columns:
- Interaction content
- Ticket Summary
- Type 2
- Type 3
- Type 4

## How to Run the Project

Step 1: Install Dependencies

    pip install -r requirements.txt

Step 2: Run the Program

    python main.py

## Output

The script prints evaluation metrics for each prediction level:
- Type 2 predictions
- Type 3 predictions (chained)
- Type 4 predictions (chained)

## Instructor Access

All contribution logs are available under GitHub Insights.
Lecturer has been added as a collaborator for full access.
