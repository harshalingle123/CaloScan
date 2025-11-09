# CaloScan - Model Integration Guide

## Overview

CaloScan now supports **two AI models** for food nutrition analysis:

### 1. **CaloScan AI v1 (GPT-2 Based)**
- **Type**: Text-to-nutrition model
- **Parameters**: 124M
- **Use Case**: When you have detailed food information (dish name, ingredients, cooking method)
- **Strengths**:
  - Flexible input with text descriptions
  - Can handle custom dishes and recipes
  - Trained on diverse nutrition data

### 2. **Food Classifier (EfficientNet-B3)**
- **Type**: Image classification + USDA nutrition lookup
- **Classes**: 101 food categories (Food-101 dataset)
- **Use Case**: Quick identification of common foods
- **Strengths**:
  - Fast and accurate food recognition
  - Uses real USDA nutrition data
  - Simple image-only workflow
  - Good for standard dishes

---

## How to Use

### Starting the Application

```bash
cd E:\Startup\CaloScan\src\web_app
python app.py
```

The application will load both models and show their status:
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

### Model Selection in UI

1. **Upload an image** of your food
2. **Select a model**:
   - **CaloScan AI v1 (GPT-2)**: For detailed analysis with custom inputs
   - **Food Classifier (EfficientNet-B3)**: For quick food recognition
3. **Add details** (optional for GPT-2, used for portion size in EfficientNet)
4. **Click "Analyze Nutrition"**

---

## Model Comparison

| Feature | GPT-2 Model | EfficientNet-B3 |
|---------|-------------|-----------------|
| Input Requirements | Dish name, ingredients (optional) | Image only |
| Food Categories | Unlimited (text-based) | 101 predefined classes |
| Nutrition Source | Model prediction | USDA FoodData Central |
| Accuracy | Good for custom dishes | High for standard foods |
| Speed | Slower (~5-10s) | Faster (~2-5s) |
| Portion Customization | Text-based | Gram-based (default 200g) |

---

## Food Categories Supported by EfficientNet-B3

The model can recognize 101 food types including:

**Desserts**: apple_pie, baklava, bread_pudding, cannoli, carrot_cake, cheesecake, chocolate_cake, chocolate_mousse, churros, creme_brulee, cup_cakes, donuts, frozen_yogurt, ice_cream, macarons, panna_cotta, red_velvet_cake, strawberry_shortcake, tiramisu, waffles

**Main Dishes**: baby_back_ribs, beef_carpaccio, beef_tartare, bibimbap, chicken_curry, chicken_quesadilla, chicken_wings, club_sandwich, crab_cakes, filet_mignon, fish_and_chips, fried_rice, grilled_salmon, hamburger, hot_dog, lasagna, lobster_roll_sandwich, pad_thai, paella, peking_duck, pho, pizza, pork_chop, prime_rib, pulled_pork_sandwich, ramen, ravioli, risotto, spaghetti_bolognese, spaghetti_carbonara, steak, sushi, tacos

**Soups & Salads**: caesar_salad, caprese_salad, clam_chowder, french_onion_soup, greek_salad, hot_and_sour_soup, miso_soup, seaweed_salad

**Appetizers & Sides**: bruschetta, deviled_eggs, dumplings, edamame, eggs_benedict, falafel, french_fries, fried_calamari, garlic_bread, gyoza, hummus, nachos, onion_rings, oysters, samosa, spring_rolls, takoyaki

And many more!

---

## USDA API Integration (Optional)

The EfficientNet model uses USDA FoodData Central for accurate nutrition data.

### To Enable USDA Integration:

1. **Get a free API key**: https://fdc.nal.usda.gov/api-key-signup
2. **Set environment variable**:
   ```bash
   # Windows
   set USDA_API_KEY=your_api_key_here

   # Linux/Mac
   export USDA_API_KEY=your_api_key_here
   ```
3. **Restart the application**

**Note**: If no API key is provided, the model uses estimated nutrition values based on food categories.

---

## File Structure

```
E:\Startup\CaloScan\
├── src/
│   ├── finetuned-food/
│   │   ├── checkpoint-3125/          # GPT-2 model
│   │   └── food_classifier_effb3/
│   │       └── food_classifier_effb3_scripted.pt  # EfficientNet model
│   └── web_app/
│       ├── app.py                    # Flask backend (updated)
│       ├── models/
│       │   └── effnet_classifier.py  # EfficientNet inference module
│       ├── templates/
│       │   └── index.html            # Frontend (updated)
│       └── static/
│           └── js/
│               └── app.js            # JavaScript (updated)
```

---

## Technical Details

### EfficientNet-B3 Pipeline

1. **Image Preprocessing**:
   - Resize to 256x256
   - Center crop to 224x224
   - Normalize with ImageNet stats

2. **Classification**:
   - TorchScript model inference
   - Returns top-5 predictions with probabilities

3. **Nutrition Lookup**:
   - Query USDA API with food name
   - Extract calories, protein, fat, carbs, fiber
   - Scale to specified portion size

### GPT-2 Pipeline

1. **Text Prompt Construction**:
   - Combine dish name, ingredients, cooking method, portion size
   - Format as instruction-following prompt

2. **Generation**:
   - Temperature: 0.7
   - Top-p: 0.9
   - Max length: 512 tokens

3. **Parsing**:
   - Extract JSON nutrition values
   - Format human-readable summary

---

## Troubleshooting

### Model Not Loading

**Issue**: "Error loading EfficientNet-B3 model"
**Solution**:
- Check model path: `E:\Startup\CaloScan\src\finetuned-food\food_classifier_effb3\food_classifier_effb3_scripted.pt`
- Ensure file exists and is not corrupted
- Check CUDA/PyTorch compatibility

### USDA API Errors

**Issue**: "Error fetching USDA data"
**Solution**:
- Verify API key is valid
- Check internet connection
- Model will fallback to estimated values automatically

### Import Errors

**Issue**: "No module named 'effnet_classifier'"
**Solution**:
- Ensure `models/effnet_classifier.py` exists
- Check Python path configuration
- Restart Flask application

---

## Future Enhancements

- [ ] Add more food classification models
- [ ] Implement MiDaS depth estimation for portion size
- [ ] Add Detectron2 for multi-food detection
- [ ] Custom nutrition database support
- [ ] Mobile model optimization (TFLite/CoreML)
- [ ] Batch processing support

---

## Credits

- **Food-101 Dataset**: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- **USDA FoodData Central**: https://fdc.nal.usda.gov/
- **EfficientNet**: https://github.com/lukemelas/EfficientNet-PyTorch
- **GPT-2**: https://huggingface.co/gpt2

---

## License

This project is part of CaloScan - AI Food Nutrition Analyzer.
For educational and research purposes.
