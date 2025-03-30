# Chained Email Classifier – Multi-Label AI Architecture

This project implements a Chained Multi-Output Architecture for classifying customer support emails into multiple labels such as Type 2, Type 3, and Type 4. 

##  Project Structure
Below is the structure of the folders and their brief description.

chained_email_classifier/
│
├── main.py                        # Main controller script
├── config.py                      # Global constants
├── requirements.txt               # Python dependencies
│
├── preprocessing/                 # Data loading and transformation
│   ├── data_loader.py
│   ├── text_vectorizer.py
│   └── label_filter.py
│
├── data_objects/                  # Encapsulation of training/test sets
│   └── encapsulated_data.py
│
├── models/                        # Base model + RandomForest
│   ├── base_model.py
│   └── random_forest_model.py
│
├── evaluation/                    # Evaluation and metrics
│   └── evaluator.py
│
└── chained_model_manager.py       # Manages chained model execution


##  Chained Multi-Output Approach

The architecture follows a chained logic:
1. Model 1 (Type 2): Predicts the primary category.
2. Model 2 (Type 3): Uses predictions from Type 2 as features.
3. Model 3 (Type 4): Uses predictions from Type 2 and Type 3 for final classification.

This chain ensures contextual accuracy across multiple levels.

##  Dataset Format

Expected CSV columns:
- `Interaction content`
- `Ticket Summary`
- `Type 2` *(label)*
- `Type 3` *(label)*
- `Type 4` *(label)*

 place the dataset file as `AppGallery.csv` in the project root.


## How to Run

### 1.  Install dependencies
```bash
pip install -r requirements.txt
```

### 2.  Run the project
```bash
python main.py
```

---

##  Output

The pipeline prints classification metrics at each stage:

- Accuracy for Type 2
- Accuracy for Type 2 + 3
- Accuracy for Type 2 + 3 + 4

---

##  Instructor Access

To access contribution metrics and commits, please check GitHub insights or contact the repository owner. Lecturer has been added as a collaborator.
