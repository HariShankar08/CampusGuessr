# Image-based Geo-localization and Orientation Prediction

This project, developed as part of the "Statistical Methods in AI" (SMAI) course at IIIT Hyderabad, explores the task of predicting geographic location (Region ID, Latitude, and Longitude) and camera orientation (Angle) from images. The project utilizes a unique dataset specifically collected and annotated by the students for this course.

## Project Context

The dataset used in this project was privately collected and manually annotated by the students taking the course. This provided us a hands-on experience in data collection, annotation, cleaning, and model development for a real-world computer vision task with unique characteristics related to spatial and angular prediction.

## Dataset

The dataset consists of images paired with corresponding metadata:

*   `filename`: The name of the image file.
*   `timestamp`: The time the image was captured.
*   `latitude`: The latitude coordinate.
*   `longitude`: The longitude coordinate.
*   `angle`: The camera's viewing angle (0-359 degrees).
*   `Region_ID`: A categorical identifier for the geographical region the image belongs to.

The data is split into training (`labels_train.csv`, `images_train`) and validation (`labels_val.csv`, `images_val`) sets. A separate test set (`images_test`) is provided for final predictions without ground truth labels.

Data cleaning steps, such as removing outliers in latitude and longitude using IQR and DBSCAN, were applied during preprocessing for some tasks.

## Project Components and Methodology

The project addresses three distinct prediction tasks using the image data:

1.  **Region ID Prediction (Image Classification):**
    *   **Goal:** Predict the categorical `Region_ID` based solely on the image content.
    *   **Approach:** Framed as a standard image classification task.
    *   **Model:** Fine-tuned a pre-trained `facebook/convnext-large-224-22k-1k` model from the Hugging Face `transformers` library. The model's classification head was adapted for the number of unique regions in the dataset.
    *   **Data:** Images were organized by `Region_ID` into directories to leverage the `datasets` library's imagefolder loading capabilities. Standard image augmentations (horizontal flips, color jitter, blur, grayscale) were applied during training using `torchvision.transforms`, followed by standard normalization.
    *   **Training:** Utilized the Hugging Face `Trainer` with mixed-precision training (`fp16`), cosine learning rate schedule, and logging to Weights & Biases (wandb).
    *   **Evaluation Metric:** Accuracy.
    *   **Validation Result:** Achieved approximately **92.41% accuracy** on the validation set using the trained model pipeline.

2.  **Latitude/Longitude Prediction (Regression with Multi-modal Input):**
    *   **Goal:** Predict the continuous `latitude` and `longitude` coordinates.
    *   **Approach:** Regression task using both image features and the predicted Region ID as input.
    *   **Model:** A custom PyTorch model (`GeoModel`) was built. It uses the pre-trained ConvNeXt as a backbone (removing its original classification head) to extract image features. It adds a linear layer to embed the one-hot encoded Region ID. The image features and region embedding are concatenated and passed through a simple regression head (`nn.Sequential`) to predict the two coordinate values.
    *   **Data:** Latitude and longitude were scaled using `StandardScaler` after outlier filtering. The Region ID was one-hot encoded. During prediction on the test set, the **Region IDs predicted by the Image Classification model were used as input**.
    *   **Transforms:** Used standard `torchvision.transforms` including resizing and center cropping, along with data augmentation for training.
    *   **Loss Function:** Mean Squared Error (MSELoss) between the predicted and true scaled coordinates.
    *   **Training:** Custom training loop with AdamW optimizer, Cosine Annealing LR, gradient clipping, early stopping based on validation loss, and wandb logging.
    *   **Evaluation:** Monitored MSE loss and Mean Absolute Angular Error (on angles derived from coordinates) during training.

3.  **Angle Prediction (Regression on Circular Data):**
    *   **Goal:** Predict the continuous `angle` (0-359 degrees).
    *   **Approach:** Regression task, treating the angle as a circular quantity by predicting its sine and cosine.
    *   **Model:** Adapted the pre-trained ConvNeXt model by replacing its classification head with a custom `AngleRegressor` that outputs two values corresponding to `[sin(angle_rad), cos(angle_rad)]`.
    *   **Data:** Angles outside the 0-359 range were filtered.
    *   **Transforms:** Used "angle-safe" augmentations (color jitter, blur, grayscale) that do not affect the intrinsic viewing angle of the image, combined with resizing and standard normalization. Random cropping and horizontal flipping were avoided.
    *   **Loss Function:** Mean Squared Error (MSELoss) between the predicted `[sin, cos]` pair and the target `[sin(angle_rad), cos(angle_rad)]` pair.
    *   **Evaluation Metric:** Mean Absolute Angular Error (MAAE), specifically designed for circular data, which correctly accounts for the 0/360 degree wrap-around.
    *   **Validation Result:** Achieved approximately **26.01 degrees MAAE** on the validation set after training for 1 epoch.

## Implementation Details

The project is implemented primarily in Python using the following key libraries:

*   `pandas`: For data loading and manipulation.
*   `torch`: For building and training neural networks.
*   `transformers`: Leveraging pre-trained vision models and utilities.
*   `datasets`: For efficient loading and handling of image datasets.
*   `torchvision`: For image loading and transformations/augmentations.
*   `scikit-learn`: For data preprocessing (scaling, outlier detection).
*   `tqdm`: For progress bars during processing and training.
*   `wandb`: For experiment tracking and visualization.
*   `PIL (Pillow)`: For image handling.

## Code Structure

The project is structured around three Jupyter notebooks, each focusing on a different prediction task:

*   `regionPred.ipynb`: Handles the Region ID classification task, including data preparation, model fine-tuning, and predicting Region IDs for the test set.
*   `regionPred-latlongPred.ipynb`: Contains both the Region ID classification (duplicated from `regionPred.ipynb` for sequential processing) and the Latitude/Longitude regression model. It includes data filtering, scaling, the custom multi-modal `GeoModel`, custom training loop, and prediction of Lat/Long using the predicted Region IDs from the first step.
*   `anglePred.ipynb`: Addresses the Angle prediction task, including data preparation, the custom `AngleRegressor` model, angle-safe transforms, custom loss and evaluation metrics, custom training loop, and prediction of angles for the test set.

## Setup and Usage

**Please Note:** This project was developed in a Kaggle environment, and the dataset is custom collected. To reproduce locally, you would need access to the same dataset files and structure, as well as potentially adapting file paths.

1.  **Obtain the dataset:** Ensure you have the `labels_train.csv`, `labels_val.csv`, `images_train/images_train`, `images_val/images_val`, and `images_test` directories available. (These are not publicly available as they were collected for the course).
2.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
3.  **Install dependencies:** It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt # (You would need to create this from the library list above)
    ```
    Alternatively, install manually:
    ```bash
    pip install pandas torch transformers datasets torchvision scikit-learn tqdm wandb Pillow
    ```
    You may also need to install `evaluate` (`pip install evaluate`) if not covered by `transformers` dependencies.
4.  **Run the notebooks:** Execute the cells in the Jupyter notebooks sequentially.
    *   Start with `regionPred.ipynb` to predict Region IDs and generate `2022114008_3.csv`.
    *   Then run `regionPred-latlongPred.ipynb` which depends on the output of the first notebook for test set predictions.
    *   Finally, run `anglePred.ipynb`.

The notebooks contain the full code for data loading, preprocessing, model definition, training, evaluation, and prediction for each task.
