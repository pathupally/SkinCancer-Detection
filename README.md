# Skin Cancer Detection Web App

This project is a deep learning pipeline and web application for classifying skin lesion images as Benign or Malignant using EfficientNet models. It includes data organization, model training, evaluation, and a Flask-based web interface for interactive predictions.

## Project Structure
- **config/**: Training configs and setup scripts
- **data/**: Raw and processed image data
- **models/**: Saved model weights
- **plots/**: Training curves and confusion matrices
- **src/**: Core Python scripts (data organization, training, utilities)
- **static/**: CSS and uploaded images for the web app
- **templates/**: HTML templates for the web interface

## Setup Instructions

1. **Install dependencies and prepare data directories**
   ```bash
   chmod +x config/setup.sh
   ./config/setup.sh
   ```
   - Installs Python requirements and creates necessary data folders.
   - Downloads ISIC images and metadata (requires `isic` CLI tool).

2. **Organize the dataset**
   ```bash
   python3 src/data_org.py
   ```
   - Splits raw images into train/val/test folders by class using metadata.

3. **Train the model**
   ```bash
   python3 src/train.py
   ```
   - Trains EfficientNet (B0, B3, B5, or fine-tune) on the organized dataset.
   - Saves best model to `models/best_model.pth` and logs results/plots.
   - Default is `b0` if no argument is given.

4. **Run the web application**
   ```bash
   python3 app.py
   ```
   - Starts the Flask web server at `http://0.0.0.0:5304`.
   - Upload images via the browser to get predictions.

## Web App Details
- **Frontend:** `templates/index.html`, `templates/result.html`, styled by `static/css/style.css`.
- **Uploads:** Images are temporarily stored in `static/uploads/` and deleted after prediction.
- **API:** Programmatic predictions available at `/api/predict` (POST with image file).

## Requirements
- Python 3.8+
- See `config/training_config.py` for full requirements list.

## Notes
- Ensure you have the ISIC archive and metadata for data organization.
- The model expects images sized according to the chosen EfficientNet variant.
- For best results, use a GPU (MPS, CUDA, or CPU fallback supported).

---

For more details, see comments in each script and the configuration options in `config/training_config.py`.
