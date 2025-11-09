"""Quick test to verify model can be loaded and generate predictions"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

print("="*80)
print("QUICK MODEL TEST")
print("="*80)

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
print("\nLoading model from checkpoint-3125...")
model_path = r"E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125"

try:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
    print(f"  Parameters: {model.num_parameters():,}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Test with a sample
print("\n" + "="*80)
print("TESTING PREDICTION")
print("="*80)

instruction = "Analyze the given food image and return the nutritional profile in JSON (Calories, Protein, Fat, Carbs, Fiber) and short human summary."
test_input = """Image URL: https://example.com/chicken.jpg
Dish: Grilled Chicken Breast
Ingredients: ["chicken breast", "olive oil", "salt", "pepper"]
Cooking method: Grilling
Portion size: ["chicken:200g"]"""

print(f"\nInput:\n{test_input}")

# Generate
prompt = f"{instruction}\n\nInput: {test_input}\n\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

print("\nGenerating prediction...")
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract output
if "Output:" in generated_text:
    prediction = generated_text.split("Output:")[-1].strip()
else:
    prediction = generated_text

print("\n" + "-"*80)
print("PREDICTION:")
print("-"*80)
print(prediction)
print("-"*80)

# Try to parse JSON
import re
try:
    json_match = re.search(r'\{[^}]+\}', prediction)
    if json_match:
        nutrition = json.loads(json_match.group(0))
        print("\n✓ Successfully parsed JSON!")
        print("\nNutrition Values:")
        for key, value in nutrition.items():
            print(f"  {key}: {value}")
    else:
        print("\n⚠ No JSON found in prediction")
except Exception as e:
    print(f"\n✗ Failed to parse JSON: {e}")

print("\n" + "="*80)
print("QUICK TEST COMPLETE!")
print("="*80)
print("\nThe model is working! You can now run the full test with:")
print("  python test_model.py")
