# CaloScan - Quick Start Guide

## 🚀 How to Run the Web App

### Method 1: Using Python Directly (Recommended)

Open Command Prompt or PowerShell and run:

```bash
cd E:\Startup\CaloScan\src\web_app
"C:\Users\Harshal\.conda\envs\torch-gpu\python.exe" start_app.py
```

### Method 2: Using Conda Environment

```bash
cd E:\Startup\CaloScan\src\web_app
conda activate torch-gpu
python start_app.py
```

### Method 3: Using the Batch File

Double-click `run.bat` in the `web_app` folder

---

## 📱 Using the Web App

1. **Wait for the server to start** - You'll see:
   ```
   ✓ Model loaded successfully!
   Open your browser and go to: http://localhost:5000
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload a food image**:
   - Click the upload area or drag & drop
   - Supported: PNG, JPG, JPEG, GIF, WEBP (max 16MB)

4. **Optionally add details** for better accuracy:
   - Dish name (e.g., "Grilled Chicken Breast")
   - Ingredients (e.g., `["chicken", "olive oil", "salt"]`)
   - Cooking method (select from dropdown)
   - Portion size (e.g., `["chicken:200g"]`)

5. **Click "Analyze Nutrition"**

6. **View results**:
   - Calories, Protein, Fat, Carbs, Fiber
   - Raw model prediction
   - Your uploaded image

---

## 🎯 Example Usage

### Simple Analysis (No Details)
Just upload an image of pizza → Get nutrition estimates

### Detailed Analysis (Better Accuracy)
Upload image + provide:
- Dish: "Margherita Pizza"
- Ingredients: `["dough", "tomato sauce", "mozzarella", "basil"]`
- Cooking: Baked
- Portion: `["pizza:250g"]`

→ Get more accurate nutrition estimates

---

## ⚠️ Important Notes

1. **Model Limitation**: The model is text-based (GPT-2), it doesn't actually "see" images
   - It uses dish names, ingredients, and cooking methods to predict nutrition
   - Providing details significantly improves accuracy

2. **First Request**: Takes 10-15 seconds (model loading)
   - Subsequent requests: ~5-10 seconds each

3. **Estimates Only**: Results are AI predictions, not precise measurements

4. **GPU Recommended**: Faster with CUDA-enabled GPU

---

## 🛠️ Troubleshooting

### Server won't start
```bash
# Check if model exists
dir "E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125"

# Check if Flask is installed
"C:\Users\Harshal\.conda\envs\torch-gpu\python.exe" -c "import flask; print('OK')"

# Install Flask if needed
"C:\Users\Harshal\.conda\envs\torch-gpu\python.exe" -m pip install flask Pillow
```

### Port 5000 already in use
Edit `start_app.py` line 32:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
```

### Can't upload images
- Check file size (must be < 16MB)
- Check file type (PNG, JPG, JPEG, GIF, WEBP only)
- Ensure `static/uploads/` folder exists

---

## 📊 Test the Model First

Before running the web app, you can test the model accuracy:

```bash
cd E:\Startup\CaloScan\notebooks
"C:\Users\Harshal\.conda\envs\torch-gpu\python.exe" test_model.py
```

This will:
- Test the model on 500 samples
- Calculate accuracy metrics (MAE, RMSE, R²)
- Generate detailed reports

---

## 📁 Files Overview

```
web_app/
├── start_app.py       ⭐ START HERE - Run this file
├── app.py             📱 Main Flask application
├── run.bat            🪟 Windows batch script
├── templates/
│   └── index.html     🌐 Web interface
├── static/
│   ├── css/style.css  🎨 Styles
│   ├── js/app.js      ⚡ JavaScript
│   └── uploads/       📁 Uploaded images (auto-created)
```

---

## ✨ Features

✅ Beautiful, modern UI
✅ Drag & drop image upload
✅ Real-time nutrition analysis
✅ Optional food details for better accuracy
✅ Animated results display
✅ Mobile responsive design
✅ GPU acceleration support

---

## 🎉 Enjoy!

Your AI-powered food nutrition analyzer is ready to use!

For more details, see `README.md`
