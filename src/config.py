import numpy as np
import math

class Config:
    def __init__(self, epochs=10, data_path="path/to/your/dataset", classifier_type="lightgbm"):
        """
        Initialize the configuration with default or user-defined values.
        Automatically sets model parameters based on classifier_type.
        """
        self.data_path = data_path
        self.classifier_type = classifier_type.lower()
        self.epochs = epochs
        self.early_stopping_rounds = 50
        self.logging_level = "INFO"
        self.distances = [1, 2, 3, 4, 5]
        self.angles = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]
        self.levels = 256  # 8-bit images

        # Set model-specific parameters
        if self.classifier_type == "xgboost":
            self.model_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": 500,
                "use_label_encoder": False,
                "learning_rate": 0.01,
                "max_depth": 6,
                "random_state": 42,
                "scale_pos_weight": 1.97,
                "early_stopping_rounds": 10,
                "tree_method": 'hist'
            }
        elif self.classifier_type == "randomforest":
            self.model_params = {
                "n_estimators": 200,
                "max_depth": 5,
                "random_state": 42,
                "class_weight": "balanced"
            }
        else:  # Default to LightGBM
            self.model_params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 5,
                "random_state": 42,
                "class_weight": "balanced"
            }

    def display(self):
        """
        Display the configuration settings.
        """
        print("Configuration Settings:")
        print(f"Dataset Path: {self.data_path}")
        print(f"Classifier Type: {self.classifier_type}")
        print(f"Model Parameters: {self.model_params}")
        print(f"Distances: {self.distances}")
        print(f"Angles: {self.angles}")
        print(f"Levels: {self.levels}")
        print(f"Number of Epochs: {self.epochs}")
        print(f"Early Stopping Rounds: {self.early_stopping_rounds}")
        print(f"Logging Level: {self.logging_level}")
