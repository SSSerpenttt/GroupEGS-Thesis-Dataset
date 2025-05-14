import os
import cv2
import numpy as np
import json
from collections import defaultdict

def load_data(data_path, include_labels=None):
    """
    Load data from the specified path, filtering ROIs by the specified labels.

    Args:
        data_path (str): Path to the dataset directory.
        include_labels (list, optional): List of labels to include. If None, include all labels.

    Returns:
        dict: A dictionary containing train, validation, and test data.
    """
    def load_split(split_path):
        images = []
        labels = []
        rois = []

        # Find the annotations file
        annotations_file = None
        for file in os.listdir(split_path):
            if file.endswith(".json"):
                annotations_file = os.path.join(split_path, file)
                break

        if not annotations_file:
            raise FileNotFoundError(f"No JSON annotations file found in {split_path}")

        # Load JSON annotations
        with open(annotations_file, "r") as f:
            data = json.load(f)

        # Map image_id to filename and category
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        id_to_category = {cat["id"]: cat["name"] for cat in data["categories"]}

        # Group annotations by image_id
        annotations_by_image = defaultdict(list)
        for ann in data["annotations"]:
            annotations_by_image[ann["image_id"]].append(ann)

        # Process each image and its annotations
        for img_id, anns in annotations_by_image.items():
            file_path = os.path.join(split_path, id_to_filename[img_id])
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image_rois = []
                image_labels = []
                for ann in anns:
                    cat_id = ann["category_id"]
                    category_name = id_to_category[cat_id]
                    # Filter by include_labels if specified
                    if include_labels is None or category_name in include_labels:
                        # Convert ROI coordinates to integers
                        x, y, w, h = [int(coord) for coord in ann["bbox"]]
                        image_rois.append([x, y, w, h])
                        image_labels.append(category_name)
                if image_rois:  # Only add images with valid ROIs
                    images.append(image)
                    rois.append(image_rois)
                    labels.append(image_labels)

        return np.array(images), labels, rois

    # Load train, validation, and test splits
    train_images, train_labels, train_rois = load_split(os.path.join(data_path, "train"))
    val_images, val_labels, val_rois = load_split(os.path.join(data_path, "valid"))
    test_images, test_labels, test_rois = load_split(os.path.join(data_path, "test"))

    return {
        "train_data": {"images": train_images, "labels": train_labels, "rois": train_rois},
        "val_data": {"images": val_images, "labels": val_labels, "rois": val_rois},
        "test_data": {"images": test_images, "labels": test_labels, "rois": test_rois},
    }
