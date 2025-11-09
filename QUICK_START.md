# CaloScan - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Test Model Loading
```bash
cd E:\Startup\CaloScan\src\web_app
python test_models.py
```

You should see:
```
✓ EfficientNet classifier module imported successfully
✓ PyTorch and Transformers imported successfully
✓ Using device: cuda
✓ GPT-2 model directory found
✓ EfficientNet model file found
✓ EfficientNet model loaded successfully!
```

### Step 2: Start the Application
```bash
python app.py
```

Wait for both models to load:
```
================================================================================
CaloScan Web App Starting...
================================================================================

Models loaded:
  - GPT-2 (CaloScan v1): ✓
  - EfficientNet-B3: ✓

Open your browser and navigate to:
  http://localhost:5000
```

### Step 3: Test Both Models

#### Test 1: EfficientNet-B3 (Recommended for quick tests)
1. Open `http://localhost:5000` in your browser
2. Upload a food image (e.g., pizza, burger, salad)
3. Select **"Food Classifier (EfficientNet-B3)"**
4. (Optional) Enter portion size like "200g" or "150g"
5. Click **"Analyze Nutrition"**
6. See results with top-5 predictions and USDA nutrition data

#### Test 2: GPT-2 Model (For detailed analysis)
1. Upload a food image
2. Select **"CaloScan AI v1 (GPT-2)"**
3. Fill in optional fields:
   - Dish Name: "Grilled Chicken Salad"
   - Ingredients: ["chicken", "lettuce", "tomato"]
   - Cooking Method: "Grilled"
   - Portion Size: ["chicken:150g", "salad:100g"]
4. Click **"Analyze Nutrition"**
5. See detailed nutrition prediction

---

## 📊 What Each Model Does

### EfficientNet-B3 🎯
**Best for**: Quick identification of common foods
- Recognizes 101 food types
- Uses real USDA nutrition data
- Fast (2-5 seconds)
- No text input needed

**Example foods**: Pizza, Burger, Sushi, Ice Cream, Caesar Salad, etc.

### GPT-2 Model 🧠
**Best for**: Custom dishes with detailed inputs
- Flexible with any food type
- Uses text descriptions
- Good for recipes and combinations
- Slower but more detailed (5-10 seconds)

**Example**: "Homemade chicken curry with rice and vegetables"

---

## ⚙️ Optional: USDA API Setup

For **real** USDA nutrition data (instead of estimates):

1. Get free API key: https://fdc.nal.usda.gov/api-key-signup
2. Set environment variable:
   ```bash
   # Windows (cmd)
   set USDA_API_KEY=your_key_here

   # Windows (PowerShell)
   $env:USDA_API_KEY="your_key_here"
   ```
3. Restart the Flask app

---

## 🎯 Quick Examples

### Example 1: Pizza (EfficientNet)
- **Model**: EfficientNet-B3
- **Upload**: Any pizza image
- **Portion**: 200g
- **Expected**: "Pizza" detected with ~90%+ confidence
- **Nutrition**: Real USDA data for pizza

### Example 2: Mixed Salad (GPT-2)
- **Model**: GPT-2
- **Upload**: Salad image
- **Dish**: "Caesar Salad"
- **Ingredients**: ["romaine lettuce", "chicken", "parmesan", "dressing"]
- **Cooking**: "Grilled"
- **Expected**: Detailed nutrition breakdown

---

## ❓ Troubleshooting

### Models not loading?
```bash
# Check if model files exist
dir E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125
dir E:\Startup\CaloScan\src\finetuned-food\food_classifier_effb3
```

### Import errors?
```bash
# Check if models directory has __init__.py
dir E:\Startup\CaloScan\src\web_app\models
```

### Port 5000 already in use?
Edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use 5001 instead
```

---

## 📚 More Information

- **Full Guide**: See `MODEL_INTEGRATION_GUIDE.md`
- **Technical Details**: See `INTEGRATION_SUMMARY.md`
- **Code Structure**: See folder structure in summary

---

## ✅ Checklist

- [ ] Ran `test_models.py` successfully
- [ ] Started Flask app with both models loaded
- [ ] Tested EfficientNet-B3 with a food image
- [ ] Tested GPT-2 model with text inputs
- [ ] (Optional) Configured USDA API key

---

**Happy Analyzing! 🍕🥗🍔**
