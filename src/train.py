from src.glcm_model import GLCMModel
from src.config import Config
from src.utils.data_loader import load_data
import joblib

def main():
    try:
        # Load configuration settings
        config = Config()
        config.display()  # Display configuration settings

        # Load data
        train_data, val_data, test_data = load_data(config.data_path)
        print("Data loaded successfully.")

        # Initialize the GLCM model
        model = GLCMModel(config)
        print("Model initialized successfully.")

        # Train the model
        print("Starting training...")
        model.train(train_data, val_data)
        print("Training completed.")

        # Save the trained model
        model_path = "trained_glcm_model.pkl"
        joblib.dump(model.model, model_path)
        print(f"Model saved to {model_path}.")

        # Evaluate the model on test data
        print("Evaluating the model on test data...")
        accuracy, report = model.evaluate(test_data)
        print(f"Test Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", report)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()