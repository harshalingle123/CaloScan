# How to Add New Models to CaloScan

This guide explains how to add additional AI models to the CaloScan web app.

## Current Setup

Currently, the app has one model:
- **Fine-Tuned Food Model** (finetuned-gpt2) - Your custom GPT-2 model

## Steps to Add a New Model

### Step 1: Update the HTML (Add Model Option)

Edit `templates/index.html` and add a new model option after the existing one:

```html
<!-- Find this section in index.html -->
<div class="model-selection">
    <!-- Existing model -->
    <label class="model-option">
        <input type="radio" name="model" value="finetuned-gpt2" checked>
        <div class="model-card">
            <div class="model-name">Fine-Tuned Food Model</div>
            <div class="model-desc">Custom trained GPT-2 (124M params) - Best for food nutrition</div>
            <div class="model-status">✅ Active</div>
        </div>
    </label>

    <!-- ADD YOUR NEW MODEL HERE -->
    <label class="model-option">
        <input type="radio" name="model" value="your-model-id">
        <div class="model-card">
            <div class="model-name">Your Model Name</div>
            <div class="model-desc">Description of your model (e.g., Vision-Language Model - Can see images)</div>
            <div class="model-status">✅ Active</div>
        </div>
    </label>

    <!-- Remove or keep the placeholder -->
    <div class="future-models-placeholder">
        <div class="placeholder-text">More models coming soon...</div>
    </div>
</div>
```

**Key fields to customize:**
- `value="your-model-id"` - Unique identifier for your model
- `model-name` - Display name shown to users
- `model-desc` - Short description of model capabilities
- `model-status` - Status indicator (✅ Active, 🔄 Loading, ⚠️ Beta, etc.)

### Step 2: Update the Backend (Load and Use the Model)

Edit `app.py`:

#### 2a. Add Model Loading

```python
# At the top of app.py, add global variables for your new model
model = None
tokenizer = None
device = None

# Add variables for your new model
your_model = None
your_tokenizer = None
# ... any other components needed

def load_model():
    """Load the fine-tuned model"""
    global model, tokenizer, device, your_model, your_tokenizer

    print("Loading fine-tuned model...")
    model_path = r"E:\Startup\CaloScan\src\finetuned-food\checkpoint-3125"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load existing model
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.to(device)
        model.eval()

        print(f"Model loaded successfully! ({model.num_parameters():,} parameters)")

        # ADD YOUR NEW MODEL LOADING HERE
        # Example for BLIP-2:
        # from transformers import Blip2Processor, Blip2ForConditionalGeneration
        # your_processor = Blip2Processor.from_pretrained("path/to/your/model")
        # your_model = Blip2ForConditionalGeneration.from_pretrained("path/to/your/model")
        # your_model.to(device)
        # your_model.eval()
        # print("Your model loaded successfully!")

        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
```

#### 2b. Create Analysis Function for Your Model

```python
def analyze_with_your_model(image_path, dish_name, ingredients, cooking_method, portion_size):
    """
    Analyze using your new model
    """
    global your_model, your_tokenizer, device

    # TODO: Implement your model's analysis logic
    # Example structure:

    # 1. Load and process the image
    # from PIL import Image
    # image = Image.open(image_path)

    # 2. Prepare inputs for your model
    # inputs = your_processor(images=image, return_tensors="pt").to(device)

    # 3. Generate predictions
    # with torch.no_grad():
    #     outputs = your_model.generate(**inputs)

    # 4. Parse and return results
    # prediction = your_processor.decode(outputs[0], skip_special_tokens=True)

    # For now, return a placeholder
    return {
        'prediction_text': 'Your model prediction will appear here',
        'nutrition': {
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbohydrates': 0,
            'fiber': 0
        }
    }
```

#### 2c. Update the Analyze Route

```python
@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle image upload and analysis"""

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP)'}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get form data
        selected_model = request.form.get('model', 'finetuned-gpt2')
        dish_name = request.form.get('dish_name', '')
        ingredients = request.form.get('ingredients', '')
        cooking_method = request.form.get('cooking_method', '')
        portion_size = request.form.get('portion_size', '')

        # SELECT THE RIGHT MODEL BASED ON USER CHOICE
        if selected_model == 'your-model-id':
            result = analyze_with_your_model(
                filepath,
                dish_name=dish_name,
                ingredients=ingredients,
                cooking_method=cooking_method,
                portion_size=portion_size
            )
        else:  # Default to finetuned-gpt2
            result = analyze_image(
                filepath,
                dish_name=dish_name,
                ingredients=ingredients,
                cooking_method=cooking_method,
                portion_size=portion_size
            )

        # Add metadata to result
        result['image_path'] = filepath.replace('\\', '/')
        result['model_used'] = selected_model

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
```

### Step 3: Test Your New Model

1. Restart the Flask server
2. Open http://localhost:5000
3. You should see your new model option appear
4. Upload an image and select your new model
5. Click "Analyze Nutrition"
6. Verify the results are generated correctly

## Example: Adding BLIP-2 (Vision-Language Model)

Here's a complete example for adding a BLIP-2 model that can actually process images:

### HTML Addition:
```html
<label class="model-option">
    <input type="radio" name="model" value="blip2">
    <div class="model-card">
        <div class="model-name">BLIP-2 Vision Model</div>
        <div class="model-desc">Vision-Language model (2.7B params) - Can actually see and analyze images</div>
        <div class="model-status">✅ Active</div>
    </div>
</label>
```

### Backend Addition:
```python
# In load_model() function:
from transformers import Blip2Processor, Blip2ForConditionalGeneration

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model.to(device)
blip_model.eval()

# New analysis function:
def analyze_with_blip2(image_path, dish_name, ingredients, cooking_method, portion_size):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    prompt = f"Analyze this food image and provide nutritional information in JSON format with calories, protein, fat, and carbohydrates. Dish: {dish_name}"

    inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = blip_model.generate(**inputs, max_length=200)

    prediction = blip_processor.decode(outputs[0], skip_special_tokens=True)

    nutrition_data = parse_nutrition_json(prediction)

    return {
        'prediction_text': prediction,
        'nutrition': nutrition_data,
        'input_data': {
            'dish_name': dish_name,
            'ingredients': ingredients,
            'cooking_method': cooking_method,
            'portion_size': portion_size
        }
    }

# Update analyze route:
if selected_model == 'blip2':
    result = analyze_with_blip2(filepath, dish_name, ingredients, cooking_method, portion_size)
elif selected_model == 'finetuned-gpt2':
    result = analyze_image(filepath, dish_name, ingredients, cooking_method, portion_size)
```

## Tips

1. **Model IDs**: Use lowercase with hyphens (e.g., `blip2`, `llava-next`, `gpt4-vision`)
2. **Descriptions**: Keep them concise but informative
3. **Status Icons**: Use emojis for visual feedback (✅ ⚠️ 🔄 🆕)
4. **Error Handling**: Always wrap model loading and inference in try-catch blocks
5. **Memory**: Consider model size - large models may require more GPU memory
6. **Default Model**: The first model with `checked` attribute is selected by default

## Checklist Before Adding a Model

- [ ] Model is downloaded/accessible
- [ ] Dependencies are installed (transformers, torch, etc.)
- [ ] Sufficient GPU/CPU memory available
- [ ] Model loading code tested separately
- [ ] Inference function returns correct format
- [ ] Error handling implemented
- [ ] UI updated with model info
- [ ] Backend routing updated
- [ ] Tested with sample images

## Future Model Ideas

Here are some models you might want to add:

1. **LLaVA** - Vision-language model, can see images
2. **GPT-4 Vision** - Via OpenAI API, best vision understanding
3. **BLIP-2** - Good balance of speed and accuracy
4. **Florence-2** - Microsoft's vision model
5. **Qwen-VL** - Efficient vision-language model
6. **Gemini Vision** - Google's multimodal model

## Need Help?

If you encounter issues:
1. Check the Flask console for error messages
2. Verify model paths are correct
3. Ensure all dependencies are installed
4. Test model loading in a separate Python script first
5. Check GPU memory usage (`nvidia-smi` on Windows/Linux)

---

**Current Models:**
- ✅ Fine-Tuned GPT-2 Food Model (checkpoint-3125)

**Ready to add more? Follow the steps above!**
