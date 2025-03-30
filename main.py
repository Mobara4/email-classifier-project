from config import TEXT_COLUMNS, LABEL_COLUMNS, MIN_SAMPLES_PER_CLASS
from preprocessing.data_loader import load_data
from preprocessing.text_vectorizer import vectorize_text
from preprocessing.label_filter import filter_labels
from data_objects.encapsulated_data import EmailDataSet
from chained_model_manager import ChainedModelManager

def main():
    print(" main.py started")
    
    # Load data
    print(" Loading data...")
    df = load_data()
    print(f"Initial dataset shape: {df.shape}")

    # Filter rare labels
    print(" Filtering rare labels...")
    df = filter_labels(df, LABEL_COLUMNS, MIN_SAMPLES_PER_CLASS)
    print(f"Filtered dataset shape: {df.shape}")

    # Vectorize text columns
    print(" Vectorizing text...")
    X = vectorize_text(df, TEXT_COLUMNS)

    # Extract labels
    y2, y3, y4 = df["Type 2"], df["Type 3"], df["Type 4"]

    # Wrap into dataset object
    print(" Preparing dataset...")
    dataset = EmailDataSet(X, y2, y3, y4)

    # Run the chained model
    print(" Running Chained Model Manager...")
    model_manager = ChainedModelManager(dataset)
    model_manager.run()

    print(" Done!")

if __name__ == "__main__":
    main()
