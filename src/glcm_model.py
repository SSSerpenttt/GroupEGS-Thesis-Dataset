import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from skimage.feature import graycomatrix, graycoprops
import cv2  # For image processing
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt  # Add this import at the top of the file
from joblib import Parallel, delayed
import random
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.callback import EarlyStopping

class GLCMModel:
    def __init__(self, config):
        """
        Initialize the GLCMModel with a configuration object.
        """
        self.config = config
        self.classifier_type = config.classifier_type
        # Classifier selection
        if self.classifier_type == "lightgbm":
            self.model = LGBMClassifier(**config.model_params)
        elif self.classifier_type == "randomforest":
            self.model = RandomForestClassifier(**config.model_params)
        elif self.classifier_type == "xgboost":
            self.model = XGBClassifier(**config.model_params)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
        
        self.data = None
        self.mlb = None  # MultiLabelBinarizer instance

        print(f"Using {self.classifier_type} as the classifier.")


    @staticmethod
    def preprocess_image(image, low_in=0, high_in=255, low_out=0, high_out=255):
        """
        Apply contrast stretching to enhance the feature differences between deep and shallow regions.
        """
        # Normalize the image to [0, 1]
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Scale the image to the desired range [low_out, high_out]
        stretched_image = (normalized_image * (high_out - low_out)) + low_out
        
        # Clip to ensure values are within the specified range
        stretched_image = np.clip(stretched_image, low_out, high_out).astype(np.uint8)
        
        return stretched_image

      

    def extract_glcm_features(self, images, rois=None):
        """
        Extract GLCM features from a list of grayscale images using parallel processing.
        Includes features from both full ROI and its center patch.
        Returns both the features and the indices of successfully processed ROIs.
        """
        def process_roi(image, roi, img_idx, roi_idx):
            x, y, w, h = map(int, roi)

            # Clip ROI coordinates to fit within the image dimensions
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = max(1, min(w, image.shape[1] - x))  # Ensure width is at least 1
            h = max(1, min(h, image.shape[0] - y))  # Ensure height is at least 1

            cropped_image = image[y:y+h, x:x+w]

            # Skip invalid ROIs
            if cropped_image.size == 0:
                return None, None

            # Helper function for GLCM feature extraction
            def compute_glcm_features(region):
                min_dim = min(region.shape[:2])
                distances = [1, max(1, min_dim // 4)]
                angles = self.config.angles
                levels = self.config.levels
                glcm = graycomatrix(
                    region,
                    distances=distances,
                    angles=angles,
                    levels=levels,
                    symmetric=True,
                    normed=True,
                )
                features = [
                    graycoprops(glcm, prop).flatten()
                    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
                ]
                return np.hstack(features)

            # Normalize and apply contrast stretching to the full ROI
            cropped_image = self.preprocess_image(cropped_image)

            # GLCM from full ROI
            full_features = compute_glcm_features(cropped_image)

            # Apply contrast stretching to the center patch
            scale = 0.3162  # sqrt(0.10), scaling the center area to 10% of the full ROI
            patch_w = max(1, int(w * scale))  # Ensure patch width is at least 1
            patch_h = max(1, int(h * scale))  # Ensure patch height is at least 1

            # Center the patch within the full ROI
            center_x = x + (w - patch_w) // 2
            center_y = y + (h - patch_h) // 2

            # Extract the center patch
            center_patch = image[center_y:center_y + patch_h, center_x:center_x + patch_w]

            # Check if the center patch is valid (non-empty and large enough)
            if center_patch.size == 0 or center_patch.shape[0] < 2 or center_patch.shape[1] < 2:
                center_features = np.zeros_like(full_features)  # fallback to zeros if invalid
            else:
                # Normalize and apply contrast stretching to the center patch
                center_patch = self.preprocess_image(center_patch)
                center_features = compute_glcm_features(center_patch)

            # Combine full ROI features with center patch features
            roi_features = np.hstack([full_features, center_features])
            return roi_features, (img_idx, roi_idx)

        # Flatten the input for parallel processing
        tasks = []
        for img_idx, image in enumerate(images):
            if rois is not None and img_idx < len(rois):
                for roi_idx, roi in enumerate(rois[img_idx]):
                    tasks.append((image, roi, img_idx, roi_idx))

        # Process ROIs in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_roi)(image, roi, img_idx, roi_idx) for image, roi, img_idx, roi_idx in tasks
        )

        # Collect features and valid ROI indices
        features = []
        valid_roi_indices = []
        for feature, valid_index in results:
            if feature is not None:
                features.append(feature)
                valid_roi_indices.append(valid_index)

        return np.array(features), valid_roi_indices



    def preprocess_labels(self, labels):
        """
        Preprocess labels for binary classification (depth-deep vs depth-shallow).
        
        Args:
            labels: List of depth labels ('depth-deep' or 'depth-shallow')
        
        Returns:
            np.ndarray: Binary labels (0 for shallow, 1 for deep)
        """
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            # Convert string labels to binary (1 for shallow, 0 for deep)
            binary_labels = np.array([1 if label == 'depth-shallow' else 0 for label in labels])
            self.mlb.fit([['depth-deep'], ['depth-shallow']])
        else:
            binary_labels = np.array([1 if label == 'depth-shallow' else 0 for label in labels])
        
        return binary_labels




    def load_data(self, data_loader):
        """
        Load data using the provided data loader.
        Extract GLCM features for each ROI and associate them with labels.
        """
        print("Loading data...")
        self.data = data_loader.load_data()

        # Extract features for training data
        print("Extracting GLCM features for training data...")
        train_images = self.data['train_images']
        train_rois = self.data['train_rois']
        train_features = []
        train_valid_indices = []
        
        for idx in tqdm(range(len(train_images)), desc="Training Data Progress"):
            features, valid_indices = self.extract_glcm_features([train_images[idx]], [train_rois[idx]])
            if len(features) > 0:
                train_features.extend(features)
                train_valid_indices.extend(valid_indices)
        
        # Ensure one-to-one mapping between features and labels
        aligned_train_labels = []
        for img_idx, roi_idx in train_valid_indices:
            if img_idx < len(self.data['train_labels']) and roi_idx < len(self.data['train_labels'][img_idx]):
                aligned_train_labels.append(self.data['train_labels'][img_idx][roi_idx])
        
        self.data['train_features'] = np.array(train_features)
        self.data['train_labels'] = np.array(aligned_train_labels)

        # Similar process for validation data
        print("Extracting GLCM features for validation data...")
        val_images = self.data['val_images']
        val_rois = self.data['val_rois']
        val_features = []
        val_valid_indices = []
        
        for idx in tqdm(range(len(val_images)), desc="Validation Data Progress"):
            features, valid_indices = self.extract_glcm_features([val_images[idx]], [val_rois[idx]])
            if len(features) > 0:
                val_features.extend(features)
                val_valid_indices.extend(valid_indices)
        
        # Align validation labels
        aligned_val_labels = []
        for img_idx, roi_idx in val_valid_indices:
            if img_idx < len(self.data['val_labels']) and roi_idx < len(self.data['val_labels'][img_idx]):
                aligned_val_labels.append(self.data['val_labels'][img_idx][roi_idx])
        
        self.data['val_features'] = np.array(val_features)
        self.data['val_labels'] = np.array(aligned_val_labels)

        # Debug information
        print(f"Training features shape: {self.data['train_features'].shape}")
        print(f"Training labels shape: {self.data['train_labels'].shape}")
        print(f"Validation features shape: {self.data['val_features'].shape}")
        print(f"Validation labels shape: {self.data['val_labels'].shape}")

        print("Data loading and feature extraction completed.")



    def train(self, train_data, val_data):
        """
        Train the model using the provided training and validation data.
        Ensures one-to-one mapping between ROIs/labels and GLCM features.
        """
        print("Extracting GLCM features for training data...")
        train_features = []
        train_labels = []
        
        # Debug: Print initial counts
        print(f"Number of training images: {len(train_data['images'])}")
        print(f"Number of training ROIs: {sum(len(rois) for rois in train_data['rois'])}")
        
        # Process each image and its ROIs
        for idx in tqdm(range(len(train_data["images"])), desc="Training GLCM Extraction"):
            image = train_data["images"][idx]
            rois = train_data["rois"][idx]
            image_labels = train_data["labels"][idx]
            
            # Extract features for all ROIs in this image
            features, valid_indices = self.extract_glcm_features([image], [rois])
            
            # Create feature-label pairs ensuring one-to-one mapping
            for feature, (_, roi_idx) in zip(features, valid_indices):
                if roi_idx < len(image_labels):
                    train_features.append(feature)
                    train_labels.append(image_labels[roi_idx])
                else:
                    print(f"Warning: Skipping ROI {roi_idx} in image {idx} - no corresponding label")

        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        
        # Debug: Print shapes before preprocessing
        print(f"Shape of train_features before preprocessing: {train_features.shape}")
        print(f"Shape of train_labels before preprocessing: {train_labels.shape}")
        
        train_labels = self.preprocess_labels(train_labels)
        
        # Debug: Print final shapes
        print(f"Final shape of train_features: {train_features.shape}")
        print(f"Final shape of train_labels: {train_labels.shape}")
        
        # Similar process for validation data
        print("Extracting GLCM features for validation data...")
        val_features = []
        val_labels = []
        
        for idx in tqdm(range(len(val_data["images"])), desc="Validation GLCM Extraction"):
            image = val_data["images"][idx]
            rois = val_data["rois"][idx]
            image_labels = val_data["labels"][idx]
            
            features, valid_indices = self.extract_glcm_features([image], [rois])
            
            for feature, (_, roi_idx) in zip(features, valid_indices):
                if roi_idx < len(image_labels):
                    val_features.append(feature)
                    val_labels.append(image_labels[roi_idx])

        val_features = np.array(val_features)
        val_labels = self.preprocess_labels(val_labels)

        print(f"Shape of val_features: {val_features.shape}")
        print(f"Shape of val_labels: {val_labels.shape}")

        print("Starting training...")
        
        classifier = self.classifier_type.lower()

        if classifier == "lightgbm":
            best_score = float("inf")
            def save_best_callback(env):
                nonlocal best_score
                current_score = env.evaluation_result_list[0][2]  # ('valid_0', 'logloss', score)
                if current_score < best_score:
                    best_score = current_score
                    print(f"📈 New best logloss: {best_score:.4f}. Saving model...")
                    self.save_model("lightgbm.best_trained-glcm_model.txt", "lightgbm_mlb.json")

            self.model.fit(
                train_features,
                train_labels,
                eval_set=[(val_features, val_labels)],
                eval_metric="logloss",
                callbacks=[
                    early_stopping(self.config.early_stopping_rounds),
                    log_evaluation(period=1),
                    save_best_callback
                ]
            )
        elif classifier == "xgboost":
            self.model.fit(
                train_features,
                train_labels,
                eval_set=[(val_features, val_labels)],
                verbose=True
            )

            # Retrieve evals_result after training
            evals_result = self.model.evals_result()

            best_iter = self.model.best_iteration
            best_score = evals_result["validation_0"]["logloss"][best_iter]
            print(f"📈 Best logloss at iteration {best_iter}: {best_score:.4f}")

            self.save_model("xgboost.best_trained-glcm_model.txt", "xgboost_mlb.json")

        elif classifier == "randomforest":
            self.model.fit(train_features, train_labels)
            self.save_model("randomforest.best_trained-glcm_model.txt", "randomforest_mlb.json")

        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")

        print("✅ Training completed.")

        # Log number of trees if available
        if hasattr(self.model, "booster_"):
            print("🌲 Number of trees in trained model:", self.model.booster_.num_trees())
        elif hasattr(self.model, "estimators_"):
            print("🌲 Number of trees in trained model:", len(self.model.estimators_))



    def predict(self, images, rois):
        """
        Make predictions using the trained model.
        The model will infer depth labels for each ROI based on GLCM features.

        Args:
            images (list or np.array): List or array of grayscale images to predict on.
            rois (list): List of ROIs for each image.

        Returns:
            dict: A dictionary containing predictions, ROIs, and images.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before making predictions.")

        print("Extracting GLCM features for the specified data...")
        input_features = []
        valid_roi_indices = []  # Track valid ROI indices
        for idx in tqdm(range(len(images)), desc="Prediction Progress"):
            features, local_indices = self.extract_glcm_features([images[idx]], [rois[idx]])
            input_features.extend(features)
            valid_roi_indices.extend([(idx, roi_idx) for (_, roi_idx) in local_indices])

        input_features = np.array(input_features)

        print(f"[Debug] Extracted input features shape: {input_features.shape}")
        print(f"[Debug] Total ROI predictions expected: {len(input_features)}")

        # Predict depth labels for all ROIs
        predictions = self.model.predict(input_features)

        # Map predictions back to their corresponding images and ROIs
        mapped_predictions = [[] for _ in range(len(images))]
        for (img_idx, roi_idx), prediction in zip(valid_roi_indices, predictions):
            mapped_predictions[img_idx].append(prediction)

        # ✅ Visualize up to 5 randomly chosen images
        sampled_image_indices = random.sample(range(len(images)), min(5, len(images)))
        for img_idx in sampled_image_indices:
            print(f"\n📷 Predictions for Image {img_idx + 1}:")
            image = images[img_idx]
            image_rois = rois[img_idx]
            predicted_labels = mapped_predictions[img_idx]

            if not predicted_labels:
                print("⚠️ No valid ROIs were processed for this image.")
                continue

            print(f"ROIs: {image_rois}")
            print(f"Predicted labels (numeric): {predicted_labels}")
            if self.mlb:
                label_names = [self.mlb.classes_[p] for p in predicted_labels]
                print(f"Predicted labels (named): {label_names}")
            else:
                label_names = [str(p) for p in predicted_labels]
                print("🔍 Note: MultiLabelBinarizer not initialized yet, showing numeric labels only.")

            # Visualization
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
            axs[0].imshow(image, cmap="gray")
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(image, cmap="gray")
            axs[1].set_title(f"Predictions for Image {img_idx + 1}")

            # Draw up to 5 ROI predictions
            sampled_rois = random.sample(list(zip(image_rois, predicted_labels)), min(5, len(predicted_labels)))
            for roi, pred_label in sampled_rois:
                x, y, w, h = map(int, roi)
                label_name = self.mlb.classes_[pred_label] if self.mlb else str(pred_label)
                axs[1].add_patch(plt.Rectangle((x, y), w, h, edgecolor="blue", facecolor="none", linewidth=1.5))
                axs[1].text(
                    x, y - 5,
                    f"Pred: {label_name}",
                    color="blue",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")
                )

            axs[1].axis("off")
            plt.tight_layout()
            plt.show()

        # Return predictions mapped to images and ROIs
        return {
            "predictions": mapped_predictions,
            "rois": rois,
            "images": images,
            "valid_roi_indices": valid_roi_indices
        }




    def evaluate(self, images, rois, labels):
        """
        Evaluate the model's performance on the specified images and ROIs.
        Displays example predictions with visual comparisons.
        Returns accuracy, classification report, and predictions.

        Args:
            images (list or np.array): List or array of grayscale images to evaluate.
            rois (list): List of ROIs for each image.
            labels (list): Ground truth labels for the ROIs.
        """
        predictions_data = self.predict(images, rois)
        predictions = predictions_data["predictions"]

        # Align ground truth labels with valid ROIs and preprocess them
        valid_roi_indices = predictions_data["valid_roi_indices"]
        original_gt_labels = [labels[img_idx][roi_idx] for img_idx, roi_idx in valid_roi_indices]

        test_labels_numerical = self.preprocess_labels(original_gt_labels)

        # Flatten predictions for evaluation
        flat_predictions = [pred for preds in predictions for pred in preds]  # For metrics
        reshaped_predictions = predictions  # Already per-image from predict()

        # Calculate accuracy
        accuracy = accuracy_score(test_labels_numerical, flat_predictions)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Classification report
        report = classification_report(test_labels_numerical, flat_predictions, target_names=self.mlb.classes_)
        print("Classification Report:\n", report)

        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels_numerical, flat_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.mlb.classes_)
        disp.plot(cmap="Blues", xticks_rotation="vertical")
        plt.title("Confusion Matrix for GLCM model ({self.classifier_type} classifier)")
        plt.show()

        # Calculate mAP
        average_precision = average_precision_score(test_labels_numerical, flat_predictions)
        print(f"Mean Average Precision (mAP): {average_precision:.2f}")
        
        for img_idx in random.sample(range(len(images)), 5):
            print(f"Visualizing predictions vs ground truth for image {img_idx + 1}...")
            image = images[img_idx]
            image_rois = rois[img_idx]
            ground_truth_labels_strings = labels[img_idx]
            predicted_labels_numerical = reshaped_predictions[img_idx]

            print(f"ROIs for image {img_idx}: {image_rois}")
            print(f"GT labels: {ground_truth_labels_strings}")
            print(f"Predictions: {predicted_labels_numerical}")

            # Create a figure with 2 subplots: original + annotated
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))

            axs[0].imshow(image)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            # Plot the image with ROIs and labels
            axs[1].imshow(image, cmap="gray")
            axs[1].set_title(f"Image {img_idx + 1}: Predictions vs Ground Truth")

            for roi_idx, (roi, gt_label_string, pred_label_numerical) in enumerate(zip(image_rois, ground_truth_labels_strings, predicted_labels_numerical)):
                x, y, w, h = map(int, roi)
                gt_label_numerical = 1 if gt_label_string == 'depth-shallow' else 0
                gt_label_name = self.mlb.classes_[gt_label_numerical]
                pred_label_name = self.mlb.classes_[pred_label_numerical]
                color = "green" if gt_label_numerical == pred_label_numerical else "red"
                axs[1].add_patch(plt.Rectangle((x, y), w, h, edgecolor=color, facecolor="none", linewidth=1))
                axs[1].text(
                    x, y - 5,
                    f"GT: {gt_label_name}\nPred: {pred_label_name}",
                    color=color,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")
              )
            axs[1].axis("off")

            plt.tight_layout()
            plt.show()

        # Return accuracy, report, and predictions
        return accuracy, report, predictions




    def save_model(self, model_path, mlb_path):
        """
        Save the full LightGBM classifier (not just booster) and MultiLabelBinarizer.
        """
        joblib.dump(self.model, model_path)
        print(f"✅ Full model saved to {model_path}")

        mlb_data = {
            "classes": self.mlb.classes_.tolist()
        }
        with open(mlb_path, 'w') as f:
            json.dump(mlb_data, f)
        print(f"✅ MultiLabelBinarizer saved to {mlb_path}")


    
    
    def load_model(self, model_path, mlb_path):
        """
        Load the full LightGBM classifier and MultiLabelBinarizer.
        """
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")

        with open(mlb_path, 'r') as f:
            mlb_data = json.load(f)
        self.mlb = MultiLabelBinarizer(classes=mlb_data["classes"])
        self.mlb.fit([])  # dummy fit to restore internal state
        print(f"✅ MultiLabelBinarizer loaded from {mlb_path}")
