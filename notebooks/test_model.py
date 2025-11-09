"""
Fine-Tuned Food Nutrition Model - Testing & Evaluation Script

This script tests the fine-tuned GPT-2 model for food nutrition analysis.
Model Location: ../src/finetuned-food/checkpoint-3125
Dataset: food_instruction_data.jsonl
"""

import torch
import json
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
warnings.filterwarnings('ignore')

print("="*80)
print("FINE-TUNED FOOD NUTRITION MODEL - TESTING")
print("="*80)
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ============================================================================
# 1. Load the Fine-Tuned Model
# ============================================================================
print("="*80)
print("1. LOADING MODEL")
print("="*80)

model_path = r"E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125"
data_path = r"E:\Startup\CaloScan\notebooks\food_instruction_data.jsonl"

print(f"Model path: {model_path}")
print(f"Loading model...")

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

model.to(device)
model.eval()

print(f"Model loaded successfully!")
print(f"Model parameters: {model.num_parameters():,}")

# ============================================================================
# 2. Load Test Data
# ============================================================================
print("\n" + "="*80)
print("2. LOADING TEST DATA")
print("="*80)

data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Total samples: {len(data)}")

# Split data - use last 10% for testing
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# ============================================================================
# 3. Define Helper Functions
# ============================================================================

def generate_nutrition_prediction(instruction, input_text, max_length=512):
    """
    Generate nutrition prediction from the model
    """
    # Format the prompt similar to training
    prompt = f"{instruction}\n\nInput: {input_text}\n\nOutput:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the output part
    if "Output:" in generated_text:
        prediction = generated_text.split("Output:")[-1].strip()
    else:
        prediction = generated_text

    return prediction


def parse_nutrition_json(text):
    """
    Extract nutrition values from JSON text
    Returns: dict with calories_kcal, protein_g, fat_g, carbohydrate_g
    """
    try:
        # Try to find JSON in the text
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            json_str = json_match.group(0)
            nutrition = json.loads(json_str)
            return {
                'calories_kcal': float(nutrition.get('calories_kcal', 0)),
                'protein_g': float(nutrition.get('protein_g', 0)),
                'fat_g': float(nutrition.get('fat_g', 0)),
                'carbohydrate_g': float(nutrition.get('carbohydrate_g', 0))
            }
    except:
        pass

    # If JSON parsing fails, return zeros
    return {
        'calories_kcal': 0.0,
        'protein_g': 0.0,
        'fat_g': 0.0,
        'carbohydrate_g': 0.0
    }


def calculate_metrics(true_values, pred_values, metric_name):
    """
    Calculate various accuracy metrics
    """
    # Remove zeros from predictions (parsing failures)
    valid_indices = pred_values > 0
    true_valid = true_values[valid_indices]
    pred_valid = pred_values[valid_indices]

    if len(pred_valid) == 0:
        return {
            'metric': metric_name,
            'total_samples': len(true_values),
            'valid_predictions': 0,
            'parsing_success_rate': 0.0,
            'mae': None,
            'rmse': None,
            'mape': None,
            'r2_like': None
        }

    mae = mean_absolute_error(true_valid, pred_valid)
    rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((true_valid - pred_valid) / (true_valid + 1e-8))) * 100

    # R² like metric (correlation-based)
    ss_res = np.sum((true_valid - pred_valid) ** 2)
    ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
    r2_like = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'metric': metric_name,
        'total_samples': len(true_values),
        'valid_predictions': len(pred_valid),
        'parsing_success_rate': len(pred_valid) / len(true_values) * 100,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2_like': r2_like
    }

# ============================================================================
# 4. Quick Test
# ============================================================================
print("\n" + "="*80)
print("3. QUICK TEST")
print("="*80)

sample = test_data[0]
print(f"Input: {sample['input'][:150]}...")
print(f"\nExpected Output: {sample['output']}")

pred = generate_nutrition_prediction(sample['instruction'], sample['input'])
print(f"\nModel Prediction: {pred}")

# ============================================================================
# 5. Evaluate Model on Test Set
# ============================================================================
print("\n" + "="*80)
print("4. EVALUATING MODEL ON TEST SET")
print("="*80)

# Test on subset for speed (or use full test set)
num_test_samples = min(500, len(test_data))  # Test on 500 samples
test_subset = test_data[:num_test_samples]

print(f"Evaluating on {num_test_samples} test samples...\n")

results = []

for i, sample in enumerate(tqdm(test_subset, desc="Testing")):
    # Generate prediction
    prediction = generate_nutrition_prediction(sample['instruction'], sample['input'])

    # Parse ground truth and prediction
    true_nutrition = parse_nutrition_json(sample['output'])
    pred_nutrition = parse_nutrition_json(prediction)

    results.append({
        'sample_id': i,
        'input': sample['input'][:100],  # First 100 chars for reference
        'true_calories': true_nutrition['calories_kcal'],
        'pred_calories': pred_nutrition['calories_kcal'],
        'true_protein': true_nutrition['protein_g'],
        'pred_protein': pred_nutrition['protein_g'],
        'true_fat': true_nutrition['fat_g'],
        'pred_fat': pred_nutrition['fat_g'],
        'true_carbs': true_nutrition['carbohydrate_g'],
        'pred_carbs': pred_nutrition['carbohydrate_g'],
        'prediction_raw': prediction,
        'ground_truth': sample['output']
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
print("\nEvaluation complete!")

# ============================================================================
# 6. Calculate Accuracy Metrics
# ============================================================================
print("\n" + "="*80)
print("5. CALCULATING ACCURACY METRICS")
print("="*80)

metrics_summary = []

for nutrient in ['calories', 'protein', 'fat', 'carbs']:
    true_col = f'true_{nutrient}'
    pred_col = f'pred_{nutrient}'

    metrics = calculate_metrics(
        results_df[true_col].values,
        results_df[pred_col].values,
        nutrient.capitalize()
    )
    metrics_summary.append(metrics)

metrics_df = pd.DataFrame(metrics_summary)

print("\n" + "="*80)
print("MODEL EVALUATION RESULTS")
print("="*80)
print(metrics_df.to_string(index=False))
print("="*80)

# ============================================================================
# 7. Detailed Analysis
# ============================================================================
print("\n" + "="*80)
print("6. DETAILED ANALYSIS")
print("="*80)

valid_results = results_df[
    (results_df['pred_calories'] > 0) &
    (results_df['pred_protein'] > 0) &
    (results_df['pred_fat'] > 0) &
    (results_df['pred_carbs'] > 0)
]

print(f"\nValid predictions: {len(valid_results)} / {len(results_df)}")
print(f"Success rate: {len(valid_results)/len(results_df)*100:.2f}%")

# Calculate error statistics
print("\n" + "-"*80)
print("ERROR STATISTICS (for valid predictions only)")
print("-"*80)

for nutrient in ['calories', 'protein', 'fat', 'carbs']:
    true_col = f'true_{nutrient}'
    pred_col = f'pred_{nutrient}'

    if len(valid_results) > 0:
        errors = np.abs(valid_results[true_col] - valid_results[pred_col])
        print(f"\n{nutrient.upper()}:")
        print(f"  Mean Error: {errors.mean():.2f}")
        print(f"  Median Error: {errors.median():.2f}")
        print(f"  Std Dev: {errors.std():.2f}")
        print(f"  Max Error: {errors.max():.2f}")
        print(f"  Min Error: {errors.min():.2f}")

# ============================================================================
# 8. Example Predictions
# ============================================================================
print("\n" + "="*80)
print("7. EXAMPLE PREDICTIONS")
print("="*80)

for i in range(min(5, len(valid_results))):
    row = valid_results.iloc[i]
    print(f"\n--- Example {i+1} ---")
    print(f"Input: {row['input']}...")
    print(f"\nGround Truth: {row['ground_truth']}")
    print(f"Prediction:   {row['prediction_raw']}")
    print(f"\nComparison:")
    print(f"  Calories: True={row['true_calories']:.1f}, Pred={row['pred_calories']:.1f}, Error={abs(row['true_calories']-row['pred_calories']):.1f}")
    print(f"  Protein:  True={row['true_protein']:.1f}g, Pred={row['pred_protein']:.1f}g, Error={abs(row['true_protein']-row['pred_protein']):.1f}g")
    print(f"  Fat:      True={row['true_fat']:.1f}g, Pred={row['pred_fat']:.1f}g, Error={abs(row['true_fat']-row['pred_fat']):.1f}g")
    print(f"  Carbs:    True={row['true_carbs']:.1f}g, Pred={row['pred_carbs']:.1f}g, Error={abs(row['true_carbs']-row['pred_carbs']):.1f}g")

# ============================================================================
# 9. Save Results
# ============================================================================
print("\n" + "="*80)
print("8. SAVING RESULTS")
print("="*80)

# Save detailed results
results_df.to_csv('model_test_results.csv', index=False)
print("✓ Detailed results saved to 'model_test_results.csv'")

# Save metrics summary
metrics_df.to_csv('model_metrics_summary.csv', index=False)
print("✓ Metrics summary saved to 'model_metrics_summary.csv'")

# Create a summary report
with open('model_evaluation_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FINE-TUNED FOOD NUTRITION MODEL - EVALUATION REPORT\n")
    f.write("="*80 + "\n\n")

    f.write(f"Model Path: {model_path}\n")
    f.write(f"Test Samples: {num_test_samples}\n")
    f.write(f"Valid Predictions: {len(valid_results)}\n")
    f.write(f"Success Rate: {len(valid_results)/len(results_df)*100:.2f}%\n\n")

    f.write("METRICS SUMMARY:\n")
    f.write("-" * 80 + "\n")
    f.write(metrics_df.to_string(index=False) + "\n")
    f.write("=" * 80 + "\n")

print("✓ Evaluation report saved to 'model_evaluation_report.txt'")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print("\nFiles generated:")
print("  - model_test_results.csv (detailed predictions)")
print("  - model_metrics_summary.csv (metrics summary)")
print("  - model_evaluation_report.txt (text report)")
