# CaloScan - AI Food Nutrition Analyzer

A web application that uses a fine-tuned GPT-2 model to analyze food images and predict nutritional values.

## Features

- 🖼️ **Image Upload** - Upload food images via drag-and-drop or file selection
- 🤖 **AI Analysis** - Fine-tuned GPT-2 model predicts nutritional values
- 📊 **Detailed Results** - See calories, protein, fat, carbs, and fiber
- 📝 **Optional Details** - Provide dish name, ingredients, cooking method for better accuracy
- 🎨 **Modern UI** - Beautiful, responsive interface

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda environment with PyTorch installed

### Setup

1. Navigate to the web app directory:
```bash
cd E:\Startup\CaloScan\src\web_app
```

2. Activate your conda environment:
```bash
conda activate torch-gpu
```

3. Install dependencies (if needed):
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Using the Batch File (Windows)

Simply double-click `run.bat` or run:
```bash
run.bat
```

### Option 2: Manual Start

```bash
conda activate torch-gpu
python app.py
```

### Option 3: Direct Python

```bash
"C:\Users\Harshal\.conda\envs\torch-gpu\python.exe" app.py
```

## Usage

1. **Start the server** using one of the methods above

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload a food image**:
   - Click the upload area or drag and drop an image
   - Supported formats: PNG, JPG, JPEG, GIF, WEBP
   - Max file size: 16MB

4. **Optional - Add food details** (helps improve accuracy):
   - Dish name (e.g., "Grilled Chicken Salad")
   - Ingredients (e.g., `["chicken", "lettuce", "tomato"]`)
   - Cooking method (select from dropdown)
   - Portion size (e.g., `["chicken:200g", "salad:150g"]`)

5. **Click "Analyze Nutrition"**

6. **View results**:
   - Nutritional values displayed in cards
   - Raw model prediction shown below
   - Input information summary

## API Endpoints

### `GET /`
Returns the main web interface

### `POST /analyze`
Analyzes a food image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `file` (required): Image file
  - `dish_name` (optional): Name of the dish
  - `ingredients` (optional): List of ingredients as string
  - `cooking_method` (optional): Cooking method
  - `portion_size` (optional): Portion size information

**Response:**
```json
{
  "prediction_text": "Raw model output...",
  "nutrition": {
    "calories": 450,
    "protein": 30,
    "fat": 15,
    "carbohydrates": 45,
    "fiber": 5
  },
  "input_data": {
    "dish_name": "Grilled Chicken Salad",
    "ingredients": "[\"chicken\", \"lettuce\"]",
    "cooking_method": "Grilled",
    "portion_size": "[\"chicken:200g\"]"
  },
  "image_path": "static/uploads/food.jpg"
}
```

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

## Project Structure

```
web_app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── run.bat               # Windows run script
├── README.md             # This file
├── templates/
│   └── index.html        # Main web page
├── static/
│   ├── css/
│   │   └── style.css     # Styles
│   ├── js/
│   │   └── app.js        # JavaScript
│   └── uploads/          # Uploaded images (auto-created)
```

## Model Information

- **Model Type**: Fine-tuned GPT-2
- **Location**: `E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125`
- **Parameters**: ~124M
- **Training Data**: 100K food images with nutrition labels
- **Note**: This is a text-only model - it uses textual descriptions (dish name, ingredients, cooking method) rather than actual image analysis

## Performance

- **First Request**: ~10-15 seconds (model loading)
- **Subsequent Requests**: ~5-10 seconds per image
- **GPU Acceleration**: Automatically used if available
- **Memory Usage**: ~2-4GB GPU memory

## Limitations

⚠️ **Important Notes:**

1. **Text-Based Predictions**: The model cannot actually "see" images - it makes predictions based on textual descriptions
2. **Estimates Only**: Predictions are estimates based on training data, not precise measurements
3. **User Input Dependent**: Accuracy improves when you provide detailed information (dish name, ingredients, etc.)
4. **Not Medical Advice**: This tool is for informational purposes only

## Troubleshooting

### Model Not Loading
- Check that the model path exists: `E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125`
- Ensure you have enough GPU/CPU memory
- Verify PyTorch and Transformers are installed correctly

### Port Already in Use
- Change the port in `app.py` (line: `app.run(debug=True, host='0.0.0.0', port=5000)`)
- Or kill the process using port 5000

### Upload Errors
- Check file size (max 16MB)
- Verify file type is supported
- Ensure write permissions for `static/uploads/` directory

## Future Improvements

- [ ] Add actual image processing with vision-language models (BLIP-2, LLaVA)
- [ ] Implement batch processing for multiple images
- [ ] Add nutrition database lookup for validation
- [ ] Create user accounts and history tracking
- [ ] Export results to PDF/CSV
- [ ] Mobile app version

## License

Copyright © 2025 CaloScan

## Support

For issues or questions, please check:
- Model training notebook: `notebooks/fine_tunning.ipynb`
- Model testing script: `notebooks/test_model.py`
- Model analysis: `notebooks/MODEL_ANALYSIS.md`
