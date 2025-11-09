# Fine-Tuned Food Nutrition Model - Analysis Report

## Overview

This document provides a comprehensive analysis of the fine-tuned GPT-2 model for food nutrition prediction.

---

## Model Information

### Base Model
- **Architecture**: GPT-2 (GPT2LMHeadModel)
- **Parameters**:
  - 12 transformer layers
  - 768 embedding dimensions
  - 12 attention heads
  - Vocabulary size: 50,257 tokens
- **Context Length**: 1024 tokens

### Fine-Tuning Details
- **Model Location**: `E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125`
- **Training Steps**: 3,125 steps
- **Epochs**: 1 complete epoch
- **Batch Size**: 8
- **Dataset**: 100,000 food instruction samples

---

## Training Performance Analysis

### Loss Progression

The training shows excellent convergence:

| Stage | Step | Loss | Gradient Norm |
|-------|------|------|---------------|
| Early | 50 | 1.2119 | 1.693 |
| Mid | 1500 | 0.4963 | 0.838 |
| Final | 3100 | 0.4863 | 0.750 |

**Key Observations:**
1. ✅ **Strong Initial Learning**: Loss dropped from 1.21 to 0.67 in first 100 steps (45% reduction)
2. ✅ **Steady Convergence**: Consistent decrease throughout training
3. ✅ **Final Loss**: 0.4863 indicates good model fit
4. ✅ **Stable Gradients**: Gradient norms decreased from 1.69 to 0.75, showing stable training

### Training Curve Analysis

```
Loss Reduction by Phase:
- Steps 0-500:     1.2119 → 0.5530 (-54.4%)
- Steps 500-1500:  0.5530 → 0.4963 (-10.3%)
- Steps 1500-3125: 0.4963 → 0.4863 (-2.0%)
```

**Interpretation:**
- The model learned the majority of patterns in the first 500 steps
- Subsequent training refined the model with diminishing returns
- No signs of overfitting (loss continued to decrease slightly)

---

## Dataset Analysis

### Data Format

The model was trained on instruction-based format:

```json
{
  "instruction": "Analyze the given food image and return the nutritional profile in JSON (Calories, Protein, Fat, Carbs, Fiber) and short human summary.",
  "input": "Image URL: https://...\nDish: Fried Chicken\nIngredients: [\"chicken\",\"breading\",\"oil\"]\nCooking method: Frying\nPortion size: [\"chicken:300g\"]",
  "output": "{\"fat_g\":25.0,\"protein_g\":30.0,\"calories_kcal\":400,\"carbohydrate_g\":15.0}"
}
```

### Dataset Statistics

- **Total Samples**: 100,000
- **Data Source**: Codatta/MM-Food-100K dataset
- **Training Split**: 90,000 samples (90%)
- **Test Split**: 10,000 samples (10%)

### Input Features
1. Image URLs
2. Dish names
3. Ingredients lists
4. Cooking methods
5. Portion sizes

### Output Format
JSON structure with:
- `calories_kcal`: Energy content
- `protein_g`: Protein in grams
- `fat_g`: Fat in grams
- `carbohydrate_g`: Carbohydrates in grams

---

## Expected Model Capabilities

Based on the training configuration and loss values, the model should be able to:

### ✅ Strong Capabilities
1. **JSON Generation**: Generate well-formatted nutrition JSON
2. **Contextual Understanding**: Use dish name, ingredients, and cooking method to estimate nutrition
3. **Reasonable Estimates**: Provide plausible nutritional values based on portion sizes

### ⚠️ Limitations
1. **No Visual Processing**: The model is text-only (GPT-2), it cannot actually "see" the images
   - Image URLs are just text tokens to the model
   - Predictions are based on textual descriptions, not actual image analysis
2. **Pattern Matching**: Model learns correlations from training data, not true nutritional science
3. **Generalization**: May struggle with rare dishes or unusual ingredient combinations
4. **Portion Accuracy**: Estimates depend on how well portion sizes are specified

---

## How to Test the Model

### Method 1: Run the Python Script

```bash
cd E:\Startup\CaloScan\notebooks
conda activate torch-gpu
python test_model.py
```

### Method 2: Run the Batch File

```bash
cd E:\Startup\CaloScan\notebooks
run_test.bat
```

### Method 3: Use the Jupyter Notebook

```bash
cd E:\Startup\CaloScan\notebooks
jupyter notebook test_model.ipynb
```

Then run all cells in the notebook.

---

## Evaluation Metrics Explained

The testing script will calculate the following metrics:

### 1. Parsing Success Rate
- Percentage of predictions that generate valid JSON
- **Expected**: 70-95% (depends on model quality)

### 2. Mean Absolute Error (MAE)
- Average absolute difference between predicted and true values
- **Lower is better**
- Example: MAE of 50 for calories means average error of ±50 kcal

### 3. Root Mean Squared Error (RMSE)
- Square root of average squared errors
- Penalizes large errors more than MAE
- **Lower is better**

### 4. Mean Absolute Percentage Error (MAPE)
- Average percentage error
- **Lower is better**
- Example: MAPE of 15% means predictions are off by 15% on average

### 5. R² Score (R-squared)
- Measure of how well predictions match true values
- Range: -∞ to 1.0
- **Higher is better**
- 1.0 = perfect predictions
- 0.0 = predictions no better than mean
- < 0 = worse than predicting the mean

---

## Expected Performance Benchmarks

Based on the training loss of 0.4863, here are rough expectations:

### Optimistic Scenario (if training went well)
- **Parsing Success Rate**: 80-95%
- **Calories MAE**: 50-100 kcal
- **Protein MAE**: 5-10g
- **Fat MAE**: 3-8g
- **Carbs MAE**: 5-15g
- **Overall R²**: 0.6-0.8

### Realistic Scenario (more likely)
- **Parsing Success Rate**: 60-80%
- **Calories MAE**: 100-200 kcal
- **Protein MAE**: 8-15g
- **Fat MAE**: 5-12g
- **Carbs MAE**: 10-20g
- **Overall R²**: 0.4-0.6

### Conservative Scenario (if model struggled)
- **Parsing Success Rate**: 40-60%
- **Calories MAE**: 200-400 kcal
- **Protein MAE**: 15-30g
- **Fat MAE**: 10-20g
- **Carbs MAE**: 20-40g
- **Overall R²**: 0.2-0.4

---

## Code Understanding

### Fine-Tuning Process (from `fine_tunning.ipynb`)

1. **Data Preparation**:
   ```python
   # Loaded MM-Food-100K dataset
   # Formatted into instruction-input-output structure
   # Saved as JSONL file
   ```

2. **Model Architecture**:
   - Base: GPT-2 (text-only language model)
   - No vision encoder (despite "Image URL" in prompts)
   - Uses text descriptions to predict nutrition

3. **Training Configuration**:
   - Batch size: 8
   - Learning rate: 5e-5 → decreased to 4.32e-7
   - Total FLOPs: 8.95 × 10¹⁵
   - Training time: ~1 epoch on 100K samples

### Testing Process (from `test_model.py`)

1. **Load Model**: Load fine-tuned checkpoint-3125
2. **Prepare Data**: Split dataset 90/10 for train/test
3. **Generate Predictions**: Use model to predict nutrition for test samples
4. **Parse Output**: Extract JSON from model outputs
5. **Calculate Metrics**: Compare predictions to ground truth
6. **Analyze Results**: Generate statistics and visualizations

---

## Recommendations

### To Improve Accuracy

1. **Add More Training Epochs**
   - Current: 1 epoch
   - Try: 3-5 epochs
   - Monitor for overfitting

2. **Tune Hyperparameters**
   - Experiment with learning rates
   - Try different batch sizes
   - Add weight decay for regularization

3. **Data Augmentation**
   - Add more diverse food examples
   - Include edge cases (very small/large portions)

4. **Use Vision-Language Model**
   - Current: Text-only GPT-2
   - Consider: BLIP-2, LLaVA, or GPT-4V
   - Actually process the images, not just URLs

### For Production Use

1. **Add Validation**
   - Check if predictions are within reasonable ranges
   - Flag unusual values for human review

2. **Error Handling**
   - Handle JSON parsing failures gracefully
   - Provide fallback estimates

3. **Ensemble Methods**
   - Combine with rule-based nutrition databases
   - Average multiple model predictions

4. **User Feedback Loop**
   - Allow users to correct predictions
   - Use corrections to fine-tune model further

---

## Next Steps

1. ✅ **Run the test script** to get actual metrics
2. ✅ **Review the results** in the generated CSV files
3. ✅ **Analyze performance** - which nutrients are predicted well?
4. ✅ **Identify failure modes** - when does the model fail?
5. ✅ **Compare to baseline** - how does it compare to simple averages?
6. ✅ **Iterate** - use insights to improve the model

---

## Files Generated by Testing

After running `test_model.py`, you'll get:

1. **model_test_results.csv**
   - Detailed predictions for each test sample
   - Columns: sample_id, input, true/pred values, raw outputs

2. **model_metrics_summary.csv**
   - Summary metrics for each nutrient
   - Columns: metric, total_samples, valid_predictions, MAE, RMSE, MAPE, R²

3. **model_evaluation_report.txt**
   - Text summary of evaluation results
   - Quick reference for model performance

---

## Conclusion

The fine-tuned GPT-2 model shows promising training dynamics with good convergence. However, it's important to note that:

- ⚠️ **This is a text-only model** - it cannot actually process images
- ✅ **It learns patterns** from dish names, ingredients, and cooking methods
- 🎯 **Actual performance** can only be determined by running the test script

Run the evaluation to get concrete accuracy metrics!

---

**Generated**: 2025-11-08
**Model**: checkpoint-3125
**Training Loss**: 0.4863
