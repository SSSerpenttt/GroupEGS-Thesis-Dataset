# GLCM Depth Classifier 🏆

A machine learning model for depth classification using **Gray-Level Co-occurrence Matrix (GLCM)** texture features and **LightGBM**.

---

## 📌 Features
- Extracts **GLCM texture features** from grayscale images.
- Trains a **GradientBoostingClassifier (LightGBM)** for depth classification.
- Supports **multi-label binarization (MLB)** for structured label encoding.
- Includes **model evaluation & visualization tools**.
- Saves and loads the model efficiently with **joblib**.

---

## Project Structure

```
glcm-classification-model
├── src
│   ├── glcm_model.py        # Main GLCM classification model implementation
│   ├── train.py             # Script to orchestrate the training process
│   ├── config.py            # Configuration settings for the model
│   └── utils
│       ├── data_loader.py   # Functions for loading and preprocessing data
│       └── metrics.py       # Functions for evaluating model performance
├── notebooks
│   └── glcm_model_colab.ipynb # Jupyter notebook for Google Colab
├── requirements.txt         # List of dependencies
├── .gitignore               # Files and directories to ignore by Git
└── README.md                # Project documentation
```

## 🔧 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/SSSerpenttt/GLCM-Classifier.git
cd GLCM-Classifier
pip install -r requirements.txt
```

Verify the installation:
```
python --version
pip list
```

## Usage

## 1.Configuration

The script uses a `Config` class to manage various settings for data loading, GLCM feature extraction, and model training.  You can adjust these parameters to suit your specific dataset and experimental needs.

### Key Parameters

* **`data_path`**:  
    * Description:  Specifies the path to the dataset directory containing the image data.
    * Default: `"path/to/your/dataset"`
    * Usage:  Modify this to point to the actual location of your image files.

* **`model_params`**:  
    * Description:  A dictionary holding the hyperparameters for the LightGBM classifier.
    * Keys:
        * `"n_estimators"`:
            * Description: The number of trees in the forest.
            * Default: `300`
        * `"learning_rate"`:
            * Description: The step size at which the model learns.
            * Default: `0.05`
        * `"max_depth"`:
            * Description: The maximum depth of the trees.
            * Default: `4`
        * `"random_state"`:
            * Description:  Seed for random number generation. Ensures reproducibility.
            * Default: `42`
        * `"class_weight"`:
            * Description: Weights associated with classes. `"balanced"` automatically adjusts weights inversely proportional to class frequencies in the input data.
            * Default: `"balanced"`

* **`distances`**:  
    * Description: A list of pixel distances used in the Gray-Level Co-occurrence Matrix (GLCM) calculation.
    * Default: `[1, 2, 3, 4, 5]`
    * Usage:  These values determine how far apart pixel pairs are when calculating texture features.

* **`angles`**:  
    * Description: A list of angles (in radians) for GLCM calculation, specifying the direction between the pixel pairs.
    * Default: `[0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]`
    * Usage:  Defines the orientations to consider when analyzing texture.

* **`levels`**:  
    * Description: The number of gray levels used to quantize the image for GLCM computation.
    * Default: `256`
    * Usage:  Typically set to 256 for 8-bit images.

* **`epochs`**:  
    * Description: The number of times the entire training dataset is passed forward and backward through the neural network during training.
    * Default: `10`

* **`early_stopping_rounds`**:  
    * Description: The number of training rounds without improvement on the validation set after which training will be stopped.
    * Default: `10`

* **`logging_level`**:  
    * Description:  The level of detail in logging messages (e.g., "INFO", "DEBUG", "WARNING").
    * Default: `"INFO"`


###   Modifying the Configuration

To customize the behavior of the scripts, modify the values within the `Config` class in `src/config.py`.  For example:

```python
from src.config import Config

config = Config()
config.data_path = "/path/to/my/images"
config.model_params["n_estimators"] = 500
config.angles = [0, np.pi/2]
```

### 2. Training
Run the training script to train the model:
```python
from glcm_model import GLCMModel
config = Config()
model = GLCMModel(config)

# Load dataset
train_data, val_data = load_data(config.data_path)["train_data"], load_data(config.data_path)["val_data"]

# Train model
model.train(train_data, val_data)
```

### 3. Google Colab
For a step-by-step guide to training the model in Google Colab:
1. Open the `notebooks/glcm_model_colab.ipynb` file in Google Colab.
2. Follow the instructions in the notebook to configure, train, and evaluate the model.

### 4. Model Saving and Evaluation
After training, you can save the trained model as `trained-glcm_model.txt` file and the MLB data as `mlb.json`. Both files are needed to be able to run the trained model. You can evaluate the model's performance on the test dataset.

#### 📂 Model Saving & Loading
- For saving the trained model:
```python
model.save_model("trained-glcm_model.txt", "mlb.json")
```
- For loading the saved model:
```python
model.load_model("trained-glcm_model.txt", "mlb.json")
```

#### ✅ Checking Model Integrity
Ensure the model and MLB files are not corrupted before loading:
```python
from google.colab import files
import joblib
import json
import os
import lightgbm as lgb

def check_model_files(model_path, mlb_path):
    """Check if the trained model and MLB file are valid before loading."""
    
    if not os.path.exists(model_path) or not os.path.exists(mlb_path):
        print(f"❌ Missing files!")
        return False

    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded. Trees: {model.booster_.num_trees()}")
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

    try:
        with open(mlb_path, 'r') as f:
            mlb_data = json.load(f)
        print(f"✅ MLB file loaded. Classes: {len(mlb_data['classes'])}")
    except Exception as e:
        print(f"❌ MLB error: {e}")
        return False

    return True
```

## Dataset Structure

The dataset should be organized as follows:
```
data_path/
├── train/
│   ├── img1.png
│   ├── img2.png
│   ├── annotations.json
├── val/
│   ├── img3.png
│   ├── img4.png
│   ├── annotations.json
└── test/
    ├── img5.png
    ├── img6.png
    ├── annotations.json
```

## Evaluation

Follow run the evaluate cell provided in the notebook. This will only work if:
1. Your newly instantiated model just finished training.
2. You created a new model instance and loaded your pre-trained weights.

### 🎯 Evaluating Pre-Trained Model

If using a pre-trained model:
```python
# Upload the trained model
from google.colab import files
uploaded_model = files.upload()
model_path = next(iter(uploaded_model))

uploaded_mlb = files.upload()
mlb_path = next(iter(uploaded_mlb))

# Load model
model.load_model(model_path, mlb_path)

# Run inference
accuracy, report, predictions = model.evaluate(test_data["images"], test_data["rois"], test_data["labels"])
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
